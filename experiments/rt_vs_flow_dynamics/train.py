from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable

import torch
from torch import Tensor, nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import yaml

from adam_atan2 import AdamATan2
from .models.common import count_trainable_parameters
from .models.flow import FlowMatchingTransformer
from .models.matched_rt import MatchedRecurrentTransformer
from .tasks.sudoku import create_sudoku_loaders
from .tasks.toy import (
    FAMILY_NAMES,
    ToyBatch,
    ToyFlow,
    ToyRT,
    generate_toy_trajectories,
    toy_path_metrics,
)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("experiment config must be a YAML mapping")
    required = {"experiment", "model", "schedule", "optimizer", "conditions", "toy", "sudoku", "trace", "analysis"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"experiment config is missing sections: {sorted(missing)}")
    expected_steps = int(config["schedule"]["micro_steps"]) * int(config["schedule"]["outer_blocks"])
    if int(config["trace"]["matched_steps"]) != expected_steps:
        raise ValueError(
            f"trace.matched_steps={config['trace']['matched_steps']} does not match the training horizon {expected_steps}"
        )
    for name, specification in config["conditions"].items():
        if specification.get("kind") not in {"rt", "flow"}:
            raise ValueError(f"condition {name!r} must have kind 'rt' or 'flow'")
    return config


def resolve_device(config: dict[str, Any], local_rank: int = 0) -> torch.device:
    requested = str(config["experiment"].get("device", "cuda"))
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; use the smoke config for CPU execution")
    return torch.device("cuda", local_rank) if requested == "cuda" else torch.device(requested)


def model_config(config: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    return dict(config["model"]) | dict(metadata)


def build_matched_model(config: dict[str, Any], metadata: dict[str, Any], condition: str) -> nn.Module:
    specification = config["conditions"][condition]
    cls = MatchedRecurrentTransformer if specification["kind"] == "rt" else FlowMatchingTransformer
    return cls(model_config(config, metadata))


def assert_parameter_parity(config: dict[str, Any], metadata: dict[str, Any]) -> int:
    rt = MatchedRecurrentTransformer(model_config(config, metadata))
    flow = FlowMatchingTransformer(model_config(config, metadata))
    rt_count, flow_count = count_trainable_parameters(rt), count_trainable_parameters(flow)
    if rt_count != flow_count:
        raise AssertionError(f"parameter mismatch: RT={rt_count:,}, Flow={flow_count:,}")
    return rt_count


def _toy_subset(batch: ToyBatch, indices: Tensor, device: torch.device) -> ToyBatch:
    return ToyBatch(*(
        getattr(batch, field)[indices].to(device)
        for field in ("context", "path", "velocity", "family")
    ))


def _straight_toy_targets(batch: ToyBatch) -> tuple[Tensor, Tensor]:
    steps = batch.path.shape[1]
    t = torch.linspace(0.0, 1.0, steps, device=batch.path.device).view(1, steps, 1)
    start, end = batch.context[:, None, :2], batch.context[:, None, 2:4]
    path = (1.0 - t) * start + t * end
    velocity = (end - start).expand(-1, steps, -1)
    return path, velocity


def _train_toy_condition(
    name: str, train_data: ToyBatch, config: dict[str, Any], seed: int, device: torch.device
) -> nn.Module:
    options = config["toy"]
    torch.manual_seed(seed)
    cls = ToyRT if name == "toy_rt" else ToyFlow
    model = cls(width=int(options["width"]), depth=int(options["depth"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(options["lr"]))
    batch_size = int(options["batch_size"])
    generator = torch.Generator(device="cpu").manual_seed(seed + 31)
    for _ in range(int(options["epochs"])):
        permutation = torch.randperm(train_data.context.shape[0], generator=generator)
        for start in range(0, len(permutation), batch_size):
            batch = _toy_subset(train_data, permutation[start:start + batch_size], device)
            if name == "toy_rt":
                prediction = model.rollout(batch.context, batch.path.shape[1])
                loss = F.mse_loss(prediction, batch.path)
            else:
                target_path, target_velocity = (
                    _straight_toy_targets(batch) if name == "flow_linear" else (batch.path, batch.velocity)
                )
                states = target_path[:, :-1].reshape(-1, 2)
                contexts = batch.context[:, None, :].expand(-1, target_path.shape[1] - 1, -1).reshape(-1, 9)
                times = torch.linspace(0.0, 1.0, target_path.shape[1], device=device)[:-1]
                times = times[None, :].expand(batch.context.shape[0], -1).reshape(-1)
                predicted_velocity = model.field(states, contexts, times)
                loss = F.mse_loss(predicted_velocity, target_velocity[:, :-1].reshape(-1, 2))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


def train_toy(config: dict[str, Any]) -> Path:
    options = config["toy"]
    output_dir = Path(config["experiment"]["output_root"]) / "toy"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config)
    all_rows: list[dict[str, object]] = []
    parity_rt = ToyRT(width=int(options["width"]), depth=int(options["depth"]))
    parity_flow = ToyFlow(width=int(options["width"]), depth=int(options["depth"]))
    if count_trainable_parameters(parity_rt) != count_trainable_parameters(parity_flow):
        raise AssertionError("toy RT and Flow parameter counts differ")
    for seed in config["experiment"]["seeds"]:
        train_data = generate_toy_trajectories(int(options["train_count"]), int(options["steps"]), int(seed))
        test_data = generate_toy_trajectories(int(options["test_count"]), int(options["steps"]), int(seed) + 10_000)
        for condition in ("toy_rt", "flow_linear", "flow_curve"):
            model = _train_toy_condition(condition, train_data, config, int(seed), device)
            with torch.inference_mode():
                predicted = model.rollout(test_data.context.to(device), int(options["steps"])).cpu()
            metrics = toy_path_metrics(predicted, test_data.path)
            for family_index, family_name in enumerate(FAMILY_NAMES):
                mask = test_data.family == family_index
                row: dict[str, object] = {"seed": int(seed), "condition": condition, "family": family_name}
                row |= {key: float(value[mask].mean()) for key, value in metrics.items()}
                all_rows.append(row)
            torch.save(model.state_dict(), output_dir / f"{condition}_seed_{seed}.pt")
    fields = list(all_rows[0])
    with (output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(all_rows)
    (output_dir / "metadata.json").write_text(json.dumps({
        "config": config, "families": FAMILY_NAMES, "conditions": ["toy_rt", "flow_linear", "flow_curve"]
    }, indent=2) + "\n")
    return output_dir


def _distributed_setup() -> tuple[int, int, int]:
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, 0
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def _set_lr(optimizer: torch.optim.Optimizer, base_lr: float, step: int, warmup: int) -> float:
    lr = base_lr * min(1.0, step / max(warmup, 1)) if warmup > 0 else base_lr
    for group in optimizer.param_groups:
        group["lr"] = torch.tensor(lr, device="cpu")
    return lr


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


@torch.inference_mode()
def evaluate_sudoku(model: nn.Module, loader: Iterable[tuple[Tensor, Tensor]], steps: int, device: torch.device) -> tuple[int, int]:
    core = _unwrap(model)
    correct = torch.zeros(2, dtype=torch.long, device=device)
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = core.rollout(inputs, steps).logits
        predictions = logits.argmax(dim=-1)
        correct[0] += (predictions == targets).all(dim=-1).sum()
        correct[1] += targets.shape[0]
    if dist.is_initialized():
        dist.reduce(correct, dst=0)
    return int(correct[0]), int(correct[1])


def train_sudoku(config: dict[str, Any], conditions: list[str]) -> Path:
    rank, world_size, local_rank = _distributed_setup()
    device = resolve_device(config, local_rank)
    schedule, optimizer_config = config["schedule"], config["optimizer"]
    output_root = Path(config["experiment"]["output_root"]) / "sudoku"
    checkpoint_root = Path(config["experiment"]["checkpoint_root"])
    if rank == 0:
        output_root.mkdir(parents=True, exist_ok=True); checkpoint_root.mkdir(parents=True, exist_ok=True)
    for seed in config["experiment"]["seeds"]:
        sudoku_config = dict(config["sudoku"]); sudoku_config["seed"] = int(seed)
        smoke = bool(sudoku_config.pop("smoke", False))
        train_loader, eval_loader, metadata = create_sudoku_loaders(
            sudoku_config, rank=rank, world_size=world_size, smoke=smoke
        )
        parameter_count = assert_parameter_parity(config, metadata)
        for condition in conditions:
            if condition not in config["conditions"]:
                raise KeyError(f"unknown condition {condition!r}")
            torch.manual_seed(int(seed)); torch.cuda.manual_seed_all(int(seed))
            model = build_matched_model(config, metadata, condition).to(device)
            if world_size > 1:
                model = DDP(model, device_ids=[local_rank], static_graph=False)
            optimizer = AdamATan2(
                model.parameters(),
                lr=torch.tensor(0.0),
                betas=(float(optimizer_config["beta1"]), float(optimizer_config["beta2"])),
                weight_decay=float(optimizer_config["weight_decay"]),
                ema=optimizer_config.get("ema"),
            )
            condition_dir = checkpoint_root / condition / f"seed_{seed}"
            if rank == 0:
                condition_dir.mkdir(parents=True, exist_ok=True)
            generator = torch.Generator(device=device.type).manual_seed(int(seed) * 1000 + rank)
            micro_steps = int(schedule["micro_steps"]); outer_blocks = int(schedule["outer_blocks"])
            total_steps = micro_steps * outer_blocks
            optimizer_steps = 0; backbone_calls = 0; started = time.perf_counter()
            logs: list[dict[str, object]] = []
            wandb_run = None
            if rank == 0 and bool(config["experiment"].get("wandb", False)):
                import wandb
                wandb_run = wandb.init(
                    project=str(config["experiment"].get("wandb_project", "rt-vs-flow-dynamics")),
                    name=f"{condition}/seed_{seed}",
                    group=str(config["experiment"].get("name", "rt_vs_flow_dynamics")),
                    config=config | {"active_condition": condition, "active_seed": int(seed)},
                )
            for epoch in range(int(schedule["epochs"])):
                sampler = getattr(train_loader, "sampler", None)
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)
                model.train()
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    state = _unwrap(model).initial_state(inputs)
                    for outer in range(outer_blocks):
                        optimizer_steps += 1; backbone_calls += micro_steps
                        lr = _set_lr(
                            optimizer, float(optimizer_config["lr"]), optimizer_steps,
                            int(optimizer_config.get("warmup_steps", 0)),
                        )
                        specification = config["conditions"][condition]
                        if specification["kind"] == "rt":
                            loss, ce, auxiliary, state, _ = model(
                                inputs, targets, state, outer * micro_steps, total_steps, micro_steps,
                                float(specification.get("curvature_lambda", 0.0)),
                            )
                            fm = torch.zeros_like(ce)
                        else:
                            times = torch.rand((micro_steps, inputs.shape[0]), generator=generator, device=device)
                            loss, fm, ce = model(
                                inputs, targets, times, float(specification.get("beta", 0.0)),
                                float(optimizer_config.get("fm_weight", 1.0)),
                                float(optimizer_config.get("ce_weight", 1.0)),
                            )
                            auxiliary = fm
                        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                optimizer.swap_ema()
                model.eval()
                correct, total = evaluate_sudoku(model, eval_loader, total_steps, device)
                if rank == 0:
                    state_dict = {key.removeprefix("module."): value for key, value in _unwrap(model).state_dict().items()}
                    torch.save(state_dict, condition_dir / f"epoch_{epoch}.pt")
                    epoch_log = {
                        "epoch": epoch, "loss": float(loss.detach()), "ce": float(ce),
                        "auxiliary": float(auxiliary), "eval_exact_match": correct / max(total, 1), "lr": lr,
                    }
                    logs.append(epoch_log)
                    if wandb_run is not None:
                        wandb_run.log(epoch_log, step=optimizer_steps)
                optimizer.swap_ema()
            elapsed = time.perf_counter() - started
            if rank == 0:
                metadata_output = {
                    "condition": condition, "seed": int(seed), "config": config,
                    "model_metadata": metadata, "parameter_count": parameter_count,
                    "optimizer_steps": optimizer_steps, "backbone_calls": backbone_calls,
                    "world_size": world_size, "wall_time_seconds": elapsed,
                }
                (condition_dir / "model_config.json").write_text(json.dumps(metadata_output, indent=2) + "\n")
                with (output_root / f"training_{condition}_seed_{seed}.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(logs[0])); writer.writeheader(); writer.writerows(logs)
                if wandb_run is not None:
                    wandb_run.finish()
            if dist.is_initialized():
                dist.barrier()
    return output_root

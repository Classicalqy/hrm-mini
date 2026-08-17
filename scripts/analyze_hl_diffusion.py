#!/usr/bin/env python
"""Collect and analyze effective H/L transport during HRM reasoning.

The script is intentionally an inference-time analysis: model parameters are fixed and
each test puzzle provides one deterministic latent trajectory. Results are written to
``results/diffusion`` by default.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import sys
from typing import Any, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

from arch.hrm import HRM
from arch.layers import Carry
from train import TrainConfig, load_module


MODEL_SPECS = (
    ("H2L6-H", 6, "h"),
    ("H2L1-H", 1, "h"),
    ("H2L6-HL", 6, "hl"),
    ("H2L6-L", 6, "l"),
)
H_CYCLES = 2
PROJECT_DIM = 32
PROJECT_SEED = 20260817
BOOTSTRAP_SAMPLES = 1000


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_config(checkpoint: Path) -> TrainConfig:
    with (checkpoint.parent / "model_config.json").open() as handle:
        return TrainConfig(**yaml.safe_load(handle))


def data_kwargs(config: TrainConfig) -> dict[str, Any]:
    kwargs = dict(config.data.__pydantic_extra__ or {})
    kwargs["augment"] = False
    kwargs["repeat"] = 1
    return kwargs


def sudoku_violations(predictions: torch.Tensor) -> torch.Tensor:
    """Count invalid zeros plus duplicate digit pairs in rows, columns, and boxes."""
    grid = predictions[:, 1:].reshape(-1, 9, 9)
    batch_size = grid.shape[0]
    digits = torch.arange(1, 10, device=grid.device)
    one_hot = grid.unsqueeze(-1) == digits

    def duplicate_pairs(counts: torch.Tensor) -> torch.Tensor:
        return (counts * (counts - 1) // 2).sum(dim=tuple(range(1, counts.ndim)))

    row_counts = one_hot.sum(dim=2)
    column_counts = one_hot.sum(dim=1)
    box_counts = (
        one_hot.reshape(batch_size, 3, 3, 3, 3, 9)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch_size, 3, 3, 9, 9)
        .sum(dim=3)
    )
    invalid = (grid == 0).sum(dim=(1, 2))
    return invalid + duplicate_pairs(row_counts) + duplicate_pairs(column_counts) + duplicate_pairs(box_counts)


def correct_margin(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = logits[:, 1:].to(torch.float32)
    targets = targets[:, 1:]
    correct = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    alternatives = logits.scatter(-1, targets.unsqueeze(-1), float("-inf")).amax(dim=-1)
    return (correct - alternatives).mean(dim=-1)


class BatchTrace:
    def __init__(self, model: HRM, input_ids: torch.Tensor, targets: torch.Tensor, projection: torch.Tensor) -> None:
        self.model = model
        self.input_ids = input_ids
        self.targets = targets
        self.projection = projection
        self.blank_mask = input_ids.eq(0)
        self.blank_mask[:, 0] = False
        self.h_states: list[torch.Tensor] = []
        self.l_states: list[torch.Tensor] = []
        self.h_predictions: list[torch.Tensor] = []
        self.h_margins: list[torch.Tensor] = []
        self.l_violations: list[torch.Tensor] = []
        self.hl_h_margins: list[torch.Tensor] = []
        self.hl_l_margins: list[torch.Tensor] = []

    def project_state(self, state: torch.Tensor) -> torch.Tensor:
        projected = state.to(torch.float32) @ self.projection
        all_tokens = projected.mean(dim=1)
        blank_count = self.blank_mask.sum(dim=1, keepdim=True).clamp_min(1)
        blank_tokens = (projected * self.blank_mask.unsqueeze(-1)).sum(dim=1) / blank_count
        return torch.cat((all_tokens, blank_tokens), dim=-1).to(torch.float16).cpu()

    def add_initial(self, carry: Carry) -> None:
        batch_size, sequence_length = self.input_ids.shape
        h_initial = carry["z_H"].view(1, 1, -1).expand(batch_size, sequence_length, -1)
        l_initial = carry["z_L"].view(1, 1, -1).expand(batch_size, sequence_length, -1)
        self.h_states.append(self.project_state(h_initial))
        self.l_states.append(self.project_state(l_initial))

    @torch.inference_mode()
    def __call__(self, event: Literal["l", "h"], z_h: torch.Tensor, z_l: torch.Tensor) -> None:
        if event == "l":
            self.l_states.append(self.project_state(z_l))
            predictions = self.model.readout_logits(z_h, z_l).argmax(dim=-1)
            self.l_violations.append(sudoku_violations(predictions).to(torch.int16).cpu())
            return

        self.h_states.append(self.project_state(z_h))
        logits = self.model.readout_logits(z_h, z_l)
        self.h_predictions.append(logits.argmax(dim=-1).to(torch.int16).cpu())
        self.h_margins.append(correct_margin(logits, self.targets).cpu())
        if self.model.readout == "hl":
            h_logits, l_logits = self.model.split_hl_readout_logits(z_h, z_l)
            self.hl_h_margins.append(correct_margin(h_logits, self.targets).cpu())
            self.hl_l_margins.append(correct_margin(l_logits, self.targets).cpu())

    def finish(self) -> dict[str, np.ndarray]:
        result = {
            "h_states": torch.stack(self.h_states, dim=1).numpy(),
            "l_states": torch.stack(self.l_states, dim=1).numpy(),
            "h_predictions": torch.stack(self.h_predictions, dim=1).numpy(),
            "h_margin": torch.stack(self.h_margins, dim=1).numpy(),
            "l_violation": torch.stack(self.l_violations, dim=1).numpy(),
        }
        if self.hl_h_margins:
            result["hl_h_margin"] = torch.stack(self.hl_h_margins, dim=1).numpy()
            result["hl_l_margin"] = torch.stack(self.hl_l_margins, dim=1).numpy()
        return result


def checkpoint_path(checkpoints_root: Path, l_cycles: int, readout: str, seed: int, epoch: int) -> Path:
    return checkpoints_root / f"H2L{l_cycles}_{readout}" / f"seed_{seed}" / f"epoch_{epoch}.pt"


@torch.inference_mode()
def collect_model(
    name: str, l_cycles: int, readout: str, checkpoint: Path, output_path: Path, sample_count: int, seed: int, epoch: int,
) -> None:
    config = load_config(checkpoint)
    arch = config.arch.__pydantic_extra__ or {}
    if arch.get("H_cycles") != H_CYCLES or arch.get("L_cycles") != l_cycles or arch.get("readout") != readout:
        raise ValueError(f"Checkpoint config does not match requested {name}: {arch}")

    create_dataloader = load_module(f"dataset.{config.data.name}@create_dataloader")
    loader, metadata = create_dataloader(
        "test_hard", config.local_batch_size, rank=0, world_size=1, **data_kwargs(config),
    )
    with torch.device("cuda"):
        model = HRM(arch | metadata)
        model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True), assign=True)
    model.eval()

    generator = torch.Generator(device="cpu").manual_seed(PROJECT_SEED)
    projection = (torch.randn(model.zH_init.shape[-1], PROJECT_DIM, generator=generator) / PROJECT_DIM**0.5).cuda()
    collected: dict[str, list[np.ndarray]] = {}
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    remaining = sample_count
    progress = tqdm(total=sample_count, desc=f"Collect {name}", unit="puzzle")
    for x, y in loader:
        if remaining == 0:
            break
        x, y = x[:remaining].cuda(), y[:remaining].cuda()
        batch_size = x.shape[0]
        trace = BatchTrace(model, x, y, projection)
        carry: Carry = model.initial_carry
        trace.add_initial(carry)
        for _ in range(config.cycles_per_data):
            carry, _ = model.forward_with_trace(carry, x, trace)
        for key, value in trace.finish().items():
            collected.setdefault(key, []).append(value)
        inputs.append(x.cpu().numpy().astype(np.int16))
        targets.append(y.cpu().numpy().astype(np.int16))
        remaining -= batch_size
        progress.update(batch_size)
    progress.close()
    if remaining:
        raise RuntimeError(f"Only collected {sample_count - remaining} of requested {sample_count} puzzles.")

    arrays = {key: np.concatenate(value, axis=0) for key, value in collected.items()}
    arrays["input_ids"] = np.concatenate(inputs, axis=0)
    arrays["targets"] = np.concatenate(targets, axis=0)
    arrays["sample_ids"] = np.arange(sample_count, dtype=np.int32)
    np.savez_compressed(output_path, **arrays)
    output_path.with_suffix(".json").write_text(json.dumps({
        "name": name,
        "checkpoint": str(checkpoint),
        "seed": seed,
        "epoch": epoch,
        "samples": sample_count,
        "outer_cycles": config.cycles_per_data,
        "h_cycles": H_CYCLES,
        "l_cycles": l_cycles,
        "readout": readout,
        "projection_dim": PROJECT_DIM,
        "projection_seed": PROJECT_SEED,
        "state_representation": "concat(mean_all_tokens(zP), mean_blank_cells(zP))",
    }, indent=2) + "\n")
    del model
    gc.collect()
    torch.cuda.empty_cache()


def np_sudoku_violations(predictions: np.ndarray) -> np.ndarray:
    grid = predictions[..., 1:].reshape(*predictions.shape[:-1], 9, 9)
    digits = np.arange(1, 10)

    def duplicate_pairs(units: np.ndarray) -> np.ndarray:
        # units: [..., number_of_units, cells_per_unit]
        counts = (units[..., None] == digits).sum(axis=-2)
        return (counts * (counts - 1) // 2).sum(axis=(-1, -2))

    rows = grid
    columns = np.swapaxes(grid, -1, -2)
    boxes = (
        grid.reshape(*grid.shape[:-2], 3, 3, 3, 3)
        .swapaxes(-3, -2)
        .reshape(*grid.shape[:-2], 9, 9)
    )
    return (grid == 0).sum(axis=(-1, -2)) + duplicate_pairs(rows) + duplicate_pairs(columns) + duplicate_pairs(boxes)


def per_puzzle_msd(states: np.ndarray, max_lag: int = 16) -> tuple[np.ndarray, np.ndarray]:
    max_lag = min(max_lag, states.shape[1] - 1)
    lags = np.arange(1, max_lag + 1)
    values = np.empty((states.shape[0], max_lag), dtype=np.float64)
    states = states.astype(np.float32)
    for index, lag in enumerate(lags):
        values[:, index] = np.square(states[:, lag:] - states[:, :-lag]).sum(axis=-1).mean(axis=1)
    return lags, values


def bootstrap_alphas(msd: np.ndarray, lags: np.ndarray, rng: np.random.Generator) -> dict[str, tuple[float, float, float]]:
    windows = {"early": lags <= 4, "late": lags >= 5}
    indices = rng.integers(0, msd.shape[0], size=(BOOTSTRAP_SAMPLES, msd.shape[0]))
    result: dict[str, tuple[float, float, float]] = {}
    for name, mask in windows.items():
        x = np.log(lags[mask])
        y = np.log(np.clip(msd[:, mask].mean(axis=0), 1e-12, None))
        point = float(np.polyfit(x, y, 1)[0])
        sampled = msd[indices][:, :, mask].mean(axis=1)
        log_sampled = np.log(np.clip(sampled, 1e-12, None))
        centered_x = x - x.mean()
        slopes = ((log_sampled - log_sampled.mean(axis=1, keepdims=True)) @ centered_x) / np.square(centered_x).sum()
        result[name] = (point, float(np.quantile(slopes, 0.025)), float(np.quantile(slopes, 0.975)))
    return result


def cosine_autocorrelation(states: np.ndarray, max_lag: int = 8) -> list[tuple[int, float]]:
    deltas = np.diff(states.astype(np.float32), axis=1)
    deltas /= np.linalg.norm(deltas, axis=-1, keepdims=True).clip(1e-12)
    result = []
    for lag in range(1, min(max_lag, deltas.shape[1] - 1) + 1):
        result.append((lag, float((deltas[:, lag:] * deltas[:, :-lag]).sum(axis=-1).mean())))
    return result


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = x.reshape(-1), y.reshape(-1)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def analyze_model(name: str, arrays: dict[str, np.ndarray], l_cycles: int) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray]]:
    targets = arrays["targets"]
    h_pred = arrays["h_predictions"]
    h_states, l_states = arrays["h_states"], arrays["l_states"]
    h_violation = np_sudoku_violations(h_pred)
    behavior: list[dict[str, object]] = []
    for step in range(h_pred.shape[1]):
        behavior.append({
            "model": name,
            "h_update": step + 1,
            "exact_match": float(np.all(h_pred[:, step, 1:] == targets[:, 1:], axis=1).mean()),
            "cell_accuracy": float((h_pred[:, step, 1:] == targets[:, 1:]).mean()),
            "hamming_distance": float((h_pred[:, step, 1:] != targets[:, 1:]).sum(axis=1).mean()),
            "constraint_violations": float(h_violation[:, step].mean()),
            "correct_margin": float(arrays["h_margin"][:, step].mean()),
        })

    rng = np.random.default_rng(PROJECT_SEED)
    msd_rows: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for state_name, states in (("H", h_states), ("L", l_states)):
        lags, msd = per_puzzle_msd(states)
        for lag, value in zip(lags, msd.mean(axis=0), strict=True):
            msd_rows.append({"model": name, "state": state_name, "metric": "msd", "lag": int(lag), "value": float(value)})
        for window, (alpha, low, high) in bootstrap_alphas(msd, lags, rng).items():
            summary.append({"model": name, "state": state_name, "metric": f"alpha_{window}", "value": alpha, "ci_low": low, "ci_high": high})
        steps = np.linalg.norm(np.diff(states.astype(np.float32), axis=1), axis=-1).reshape(-1)
        for quantile in (0.5, 0.9, 0.95, 0.99):
            summary.append({"model": name, "state": state_name, "metric": f"step_q{int(quantile * 100)}", "value": float(np.quantile(steps, quantile)), "ci_low": "", "ci_high": ""})
        for lag, value in cosine_autocorrelation(states):
            msd_rows.append({"model": name, "state": state_name, "metric": "direction_autocorrelation", "lag": lag, "value": value})

    coupling: list[dict[str, object]] = []
    h_delta = np.diff(h_states.astype(np.float32), axis=1)
    l_delta = np.diff(l_states.astype(np.float32), axis=1)
    l_violation = arrays["l_violation"].astype(np.float32)
    for h_index in range(1, h_delta.shape[1]):
        start = h_index * l_cycles
        if start >= l_delta.shape[1]:
            break
        h_jump = np.linalg.norm(h_delta[:, h_index - 1], axis=-1)
        first_l = l_delta[:, start]
        first_l_norm = np.linalg.norm(first_l, axis=-1)
        alignment = (h_delta[:, h_index - 1] * first_l).sum(axis=-1) / np.clip(h_jump * first_l_norm, 1e-12, None)
        end = min(start + l_cycles - 1, l_violation.shape[1] - 1)
        reduction = l_violation[:, start] - l_violation[:, end]
        coupling.append({
            "model": name,
            "h_update": h_index + 1,
            "h_jump_norm": float(h_jump.mean()),
            "first_l_norm": float(first_l_norm.mean()),
            "h_to_l_alignment": float(alignment.mean()),
            "l_block_violation_reduction": float(reduction.mean()),
            "jump_reduction_pearson": pearson(h_jump, reduction),
        })
    return behavior, msd_rows, summary + coupling, {"h_violation": h_violation}


def plot_behavior(output_dir: Path, behavior: list[dict[str, object]]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for model in {str(row["model"]) for row in behavior}:
        rows = [row for row in behavior if row["model"] == model]
        x = [int(row["h_update"]) for row in rows]
        axes[0].plot(x, [float(row["exact_match"]) for row in rows], label=model)
        axes[1].plot(x, [float(row["hamming_distance"]) for row in rows], label=model)
        axes[2].plot(x, [float(row["constraint_violations"]) for row in rows], label=model)
    for axis, title in zip(axes, ("Exact match", "Hamming distance", "Constraint violations"), strict=True):
        axis.set_title(title); axis.set_xlabel("H update"); axis.grid(alpha=.25)
    axes[0].legend(); figure.tight_layout()
    for suffix in ("png", "pdf"): figure.savefig(output_dir / f"behavior.{suffix}", dpi=200)
    plt.close(figure)


def plot_msd(output_dir: Path, rows: list[dict[str, object]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for state, axis in zip(("H", "L"), axes, strict=True):
        for model in sorted({str(row["model"]) for row in rows}):
            data = [row for row in rows if row["model"] == model and row["state"] == state and row["metric"] == "msd"]
            axis.loglog([row["lag"] for row in data], [row["value"] for row in data], marker="o", label=model)
        axis.set_title(f"{state} intrinsic-clock MSD"); axis.set_xlabel("lag"); axis.set_ylabel("MSD"); axis.grid(alpha=.25, which="both")
    axes[0].legend(fontsize=8); figure.tight_layout()
    for suffix in ("png", "pdf"): figure.savefig(output_dir / f"msd.{suffix}", dpi=200)
    plt.close(figure)


def plot_step_and_autocorrelation(output_dir: Path, rows: list[dict[str, object]], loaded: dict[str, dict[str, np.ndarray]]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for state, axis in zip(("H", "L"), axes[0], strict=True):
        key = "h_states" if state == "H" else "l_states"
        for model, arrays in sorted(loaded.items()):
            steps = np.linalg.norm(np.diff(arrays[key].astype(np.float32), axis=1), axis=-1).reshape(-1)
            sorted_steps = np.sort(steps)
            sample = np.linspace(0, len(sorted_steps) - 1, min(400, len(sorted_steps)), dtype=int)
            axis.loglog(sorted_steps[sample], 1 - sample / len(sorted_steps), label=model)
        axis.set_title(f"{state} step-size CCDF"); axis.set_xlabel("step norm"); axis.set_ylabel("P(step ≥ x)"); axis.grid(alpha=.25, which="both")
    for state, axis in zip(("H", "L"), axes[1], strict=True):
        for model in sorted({str(row["model"]) for row in rows}):
            data = [row for row in rows if row["model"] == model and row["state"] == state and row["metric"] == "direction_autocorrelation"]
            axis.plot([row["lag"] for row in data], [row["value"] for row in data], marker="o", label=model)
        axis.axhline(0, color="black", linewidth=.8); axis.set_title(f"{state} direction autocorrelation"); axis.set_xlabel("lag"); axis.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8); figure.tight_layout()
    for suffix in ("png", "pdf"): figure.savefig(output_dir / f"step_autocorrelation.{suffix}", dpi=200)
    plt.close(figure)


def plot_coupling(output_dir: Path, rows: list[dict[str, object]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for model in sorted({str(row["model"]) for row in rows}):
        data = [row for row in rows if row["model"] == model]
        x = [row["h_update"] for row in data]
        axes[0].plot(x, [row["h_to_l_alignment"] for row in data], marker="o", label=model)
        axes[1].plot(x, [row["l_block_violation_reduction"] for row in data], marker="o", label=model)
    axes[0].axhline(0, color="black", linewidth=.8); axes[0].set_title("H→next-L alignment")
    axes[1].set_title("Next L-block violation reduction")
    for axis in axes: axis.set_xlabel("H update"); axis.grid(alpha=.25)
    axes[0].legend(fontsize=8); figure.tight_layout()
    for suffix in ("png", "pdf"): figure.savefig(output_dir / f"block_coupling.{suffix}", dpi=200)
    plt.close(figure)


def plot_hl_readout(output_dir: Path, arrays: dict[str, np.ndarray]) -> None:
    if "hl_h_margin" not in arrays:
        return
    figure, axis = plt.subplots(figsize=(6, 4))
    x = np.arange(1, arrays["hl_h_margin"].shape[1] + 1)
    axis.plot(x, arrays["hl_h_margin"].mean(axis=0), label="H logit-margin contribution")
    axis.plot(x, arrays["hl_l_margin"].mean(axis=0), label="L logit-margin contribution")
    axis.plot(x, arrays["h_margin"].mean(axis=0), label="combined margin", linewidth=2)
    axis.set_xlabel("H update"); axis.set_ylabel("Correct-token margin"); axis.grid(alpha=.25); axis.legend(fontsize=8)
    figure.tight_layout()
    for suffix in ("png", "pdf"): figure.savefig(output_dir / f"hl_readout_contribution.{suffix}", dpi=200)
    plt.close(figure)


def write_readme(output_dir: Path) -> None:
    (output_dir / "README.md").write_text("""# HRM effective-diffusion analysis

`trajectory_*.npz` stores compressed inference trajectories. `h_states` and
`l_states` have shape `[puzzle, intrinsic update including initial state, 64]`; 64 is
the concatenation of 32-dimensional projected all-token and blank-cell means.

`behavior_timeseries.csv` is measured after H updates. `msd_bootstrap.csv` contains
intrinsic-clock MSD points and direction autocorrelations; `summary_metrics.csv`
contains bootstrap MSD exponents and step quantiles. `block_coupling.csv` measures a
H jump's relation to the following L block. These are effective deterministic latent
transport statistics, not a claim that inference is stochastic diffusion.
""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-root", type=Path, default=Path("checkpoints/h2_l_readout_sweep"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/diffusion"))
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=19)
    parser.add_argument("--analyze-only", action="store_true", help="Reuse existing trajectory NPZs.")
    args = parser.parse_args()
    if args.samples <= 0: parser.error("--samples must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    for name, l_cycles, readout in MODEL_SPECS:
        checkpoint = checkpoint_path(args.checkpoints_root, l_cycles, readout, args.seed, args.epoch)
        if not args.analyze_only and not checkpoint.is_file(): missing.append(checkpoint)
        trajectory = args.output_dir / f"trajectory_{name}.npz"
        if args.analyze_only and not trajectory.is_file(): missing.append(trajectory)
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(map(str, missing)))

    all_behavior: list[dict[str, object]] = []
    all_msd: list[dict[str, object]] = []
    all_summary: list[dict[str, object]] = []
    all_coupling: list[dict[str, object]] = []
    loaded: dict[str, dict[str, np.ndarray]] = {}
    model_progress = tqdm(total=len(MODEL_SPECS), desc="Models", unit="model")
    for name, l_cycles, readout in MODEL_SPECS:
        model_progress.set_postfix_str(name)
        trajectory = args.output_dir / f"trajectory_{name}.npz"
        if not args.analyze_only:
            collect_model(name, l_cycles, readout, checkpoint_path(args.checkpoints_root, l_cycles, readout, args.seed, args.epoch), trajectory, args.samples, args.seed, args.epoch)
        with np.load(trajectory) as data:
            arrays = {key: data[key] for key in data.files}
        loaded[name] = arrays
        behavior, msd, summary_and_coupling, _ = analyze_model(name, arrays, l_cycles)
        all_behavior.extend(behavior); all_msd.extend(msd)
        all_summary.extend([row for row in summary_and_coupling if "metric" in row])
        all_coupling.extend([row for row in summary_and_coupling if "h_update" in row])
        model_progress.update(1)

    model_progress.close()
    print("Writing diffusion summary tables and figures...")
    write_csv(args.output_dir / "behavior_timeseries.csv", all_behavior, list(all_behavior[0]))
    write_csv(args.output_dir / "msd_bootstrap.csv", all_msd, list(all_msd[0]))
    write_csv(args.output_dir / "summary_metrics.csv", all_summary, ["model", "state", "metric", "value", "ci_low", "ci_high"])
    write_csv(args.output_dir / "block_coupling.csv", all_coupling, list(all_coupling[0]))
    native = [row for row in all_behavior if int(row["h_update"]) == max(int(other["h_update"]) for other in all_behavior if other["model"] == row["model"])]
    write_csv(args.output_dir / "native_performance.csv", native, list(native[0]))
    plot_behavior(args.output_dir, all_behavior)
    plot_msd(args.output_dir, all_msd)
    plot_step_and_autocorrelation(args.output_dir, all_msd, loaded)
    plot_coupling(args.output_dir, all_coupling)
    plot_hl_readout(args.output_dir, loaded["H2L6-HL"])
    write_readme(args.output_dir)


if __name__ == "__main__":
    main()

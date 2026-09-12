"""Test-time scaling for the k=55 Sudoku easy/hard experiments.

This evaluates the six trained conditions (easy/hard x HRM/TRM/RT) without
changing any checkpoint weights.  HRM and TRM run with H=2 and a swept L
budget.  RT has no H/L hierarchy; its four-layer recurrent core uses
``cycles=L+1`` so that it has the same Transformer-layer-call budget as H2L.
"""

import argparse
import csv
import importlib
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
import tqdm

from arch.layers import Carry


L_VALUES = (6, 8, 16, 32, 64, 128, 256)
CONDITIONS = (
    ("easy_k55_hrm", "hrm", "easy"),
    ("hard_k55_hrm", "hrm", "hard"),
    ("easy_k55_trm", "trm", "easy"),
    ("hard_k55_trm", "trm", "hard"),
    ("easy_k55_rt", "rt", "easy"),
    ("hard_k55_rt", "rt", "hard"),
)
CONDITION_NAMES = tuple(condition for condition, _, _ in CONDITIONS)
EXPECTED_ARCHITECTURES = {"hrm": "HRM", "trm": "TRM", "rt": "RecurrentTransformer"}


@dataclass(frozen=True)
class Checkpoint:
    condition: str
    model: str
    train_band: str
    seed: int
    path: Path
    config: dict[str, Any]


@dataclass(frozen=True)
class Run:
    condition: str
    model: str
    train_band: str
    seed: int
    directory: Path
    config: dict[str, Any]
    steps_per_epoch: int


def parse_l_values(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("--l-values must be a non-empty comma-separated list of positive integers")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("--l-values must not contain duplicates")
    return values


def load_module(identifier: str):
    module_path, class_name = identifier.split("@")
    return getattr(importlib.import_module(module_path), class_name)


def load_condition_config(condition: str) -> dict[str, Any]:
    """Compose the committed experiment config instead of deserializing it from a checkpoint."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config_dir = Path(__file__).with_name("config").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name=condition)
    result = OmegaConf.to_container(config, resolve=True)
    if not isinstance(result, dict):
        raise ValueError(f"invalid Hydra configuration for {condition}")
    return result


def steps_per_epoch_from_metadata(run_dir: Path) -> int:
    """Extract the scalar needed to map W&B steps to saved epoch filenames.

    Older train.py versions wrote OmegaConf objects with cyclic parent pointers
    into model_config.json. Parsing the whole YAML is therefore impossible, but
    this top-level run_metadata scalar is emitted as ordinary YAML.
    """
    metadata_path = run_dir / "model_config.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing checkpoint metadata: {metadata_path}")
    contents = metadata_path.read_text()
    match = re.search(r"^  steps_per_epoch_per_rank:\s*(\d+)\s*$", contents, flags=re.MULTILINE)
    if match is None or int(match.group(1)) <= 0:
        raise ValueError(f"missing positive run_metadata.steps_per_epoch_per_rank in {metadata_path}")
    return int(match.group(1))


def discover_runs(checkpoint_root: Path, seeds: Iterable[int], conditions: Iterable[str]) -> list[Run]:
    runs: list[Run] = []
    selected_conditions = set(conditions)
    condition_configs = {condition: load_condition_config(condition) for condition in selected_conditions}
    for condition, model, train_band in CONDITIONS:
        if condition not in selected_conditions:
            continue
        for seed in seeds:
            seed_dir = checkpoint_root / condition / f"seed_{seed}"
            config_path = seed_dir / "model_config.json"
            if not config_path.is_file():
                raise FileNotFoundError(f"missing run metadata for {condition}, seed {seed}: {config_path}")
            config = condition_configs[condition]
            architecture = str(config["arch"]["name"]).rsplit("@", 1)[-1]
            if architecture != EXPECTED_ARCHITECTURES[model]:
                raise ValueError(f"{seed_dir} contains {architecture}, expected {EXPECTED_ARCHITECTURES[model]}")
            runs.append(Run(
                condition, model, train_band, seed, seed_dir, config,
                steps_per_epoch_from_metadata(seed_dir),
            ))
    return runs


def best_epoch_from_history(history: Iterable[dict[str, Any]], metric: str, steps_per_epoch: int) -> tuple[int, int, float]:
    """Return epoch, W&B step, and value for a metric's best logged epoch."""
    if steps_per_epoch <= 0:
        raise ValueError("checkpoint metadata must contain a positive steps_per_epoch_per_rank")
    best: tuple[int, int, float] | None = None
    for row in history:
        try:
            step = int(row["_step"])
            value = float(row[metric])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(value) or step <= 0 or step % steps_per_epoch != 0:
            continue
        epoch = step // steps_per_epoch - 1
        candidate = (epoch, step, value)
        if best is None or value > best[2] or (value == best[2] and epoch < best[0]):
            best = candidate
    if best is None:
        raise ValueError(f"W&B history contains no epoch-aligned finite values for {metric}")
    return best


def select_best_checkpoints(runs: list[Run], entity: str, metric: str) -> tuple[list[Checkpoint], list[dict[str, Any]]]:
    """Select one saved checkpoint per run using its W&B evaluation history."""
    import wandb

    projects = {run.config.get("project_name") for run in runs}
    if len(projects) != 1 or None in projects:
        raise ValueError("all runs must define the same W&B project_name")
    project = str(projects.pop())
    api = wandb.Api(timeout=60)
    remote_runs = list(api.runs(f"{entity}/{project}", per_page=100))

    checkpoints: list[Checkpoint] = []
    selections: list[dict[str, Any]] = []
    for run in runs:
        matches = [
            remote for remote in remote_runs
            if remote.state == "finished"
            and remote.group == run.condition
            and str(remote.config.get("seed")) == str(run.seed)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one W&B run for {run.condition}, seed {run.seed}; found {len(matches)}. "
                "Use a unique MLP_TASK_NAME or remove duplicate W&B runs before evaluating."
            )
        remote = matches[0]
        epoch, step, score = best_epoch_from_history(
            remote.scan_history(keys=["_step", metric]), metric, run.steps_per_epoch
        )
        checkpoint_path = run.directory / f"epoch_{epoch}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"W&B selected epoch {epoch}, but its checkpoint is missing: {checkpoint_path}")
        checkpoints.append(Checkpoint(run.condition, run.model, run.train_band, run.seed, checkpoint_path, run.config))
        selections.append({
            "condition": run.condition,
            "model": run.model,
            "train_band": run.train_band,
            "seed": run.seed,
            "selection_metric": metric,
            "selection_exact_match": score,
            "wandb_step": step,
            "selected_epoch": epoch,
            "checkpoint": str(checkpoint_path),
            "wandb_run_id": remote.id,
            "wandb_run_name": remote.name,
        })
    return checkpoints, selections


def architecture_at_l(config: dict[str, Any], model: str, l_cycles: int) -> dict[str, Any]:
    """Return architecture options with only the inference recursion budget changed."""
    arch = dict(config["arch"])
    if model in {"hrm", "trm"}:
        if int(arch.get("H_cycles", -1)) != 2:
            raise ValueError(f"test-time scaling requires H_cycles=2, got {arch.get('H_cycles')}")
        arch["L_cycles"] = l_cycles
    elif model == "rt":
        # H2L has 4 * (L + 1) Transformer-layer calls: two H rounds and
        # two-layer H/L cores. RT has a four-layer core, hence cycles=L+1.
        arch["cycles"] = l_cycles + 1
    else:
        raise ValueError(f"unsupported model: {model}")
    return arch


def evaluation_spec(config: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, dict[str, Any]]]:
    """Extract the shared held-out easy/hard loaders from one run config."""
    raw_data = config.get("data")
    if not isinstance(raw_data, dict) or not isinstance(raw_data.get("name"), str):
        raise ValueError("checkpoint must define a named data configuration")
    data_name = raw_data["name"]
    data_kwargs = {key: value for key, value in raw_data.items() if key != "name"}
    eval_sets = data_kwargs.pop("eval_sets", None)
    if not isinstance(eval_sets, dict) or set(eval_sets) != {"easy", "hard"}:
        raise ValueError("checkpoint must define exactly easy and hard evaluation sets")
    expected = {
        "easy": {"eval_blank_max": 55},
        "hard": {"eval_blank_min": 56},
    }
    for band, bounds in expected.items():
        options = dict(eval_sets[band])
        if options.get("split", "test") != "test" or any(options.get(key) != value for key, value in bounds.items()):
            raise ValueError(f"{band} evaluation set must be the k=55 held-out test band")
    return data_name, data_kwargs, {band: dict(options) for band, options in eval_sets.items()}


def validate_shared_evaluation(checkpoints: list[Checkpoint]) -> tuple[str, dict[str, Any], dict[str, dict[str, Any]]]:
    reference_name, reference_data, reference_sets = evaluation_spec(checkpoints[0].config)
    keys = ("dataset_name", "eval_dataset_name", "eval_num_base_puzzles", "eval_seed")
    reference_values = {key: reference_data.get(key) for key in keys}
    for checkpoint in checkpoints[1:]:
        data_name, data, eval_sets = evaluation_spec(checkpoint.config)
        if (
            data_name != reference_name
            or {key: data.get(key) for key in keys} != reference_values
            or eval_sets != reference_sets
        ):
            raise ValueError(f"evaluation protocol differs for {checkpoint.path}")
    return reference_name, reference_data, reference_sets


def create_eval_loaders(
    data_name: str, data_kwargs: dict[str, Any], eval_sets: dict[str, dict[str, Any]], batch_size: int
) -> dict[str, Any]:
    create_dataloader = load_module(f"dataset.{data_name}@create_dataloader")
    loaders = {}
    for band, options in eval_sets.items():
        loader_kwargs = data_kwargs | options
        split = loader_kwargs.pop("split", "test")
        loaders[band] = create_dataloader(
            split,
            batch_size,
            rank=0,
            world_size=1,
            drop_last=False,
            **loader_kwargs,
        )[0]
    return loaders


@torch.inference_mode()
def evaluate_model(model: nn.Module, loader: Any, cycles_per_data: int, label: str) -> tuple[int, float]:
    total_correct = 0
    total_samples = 0
    for x, y in tqdm.tqdm(loader, desc=label, leave=False):
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        carry: Carry = model.initial_carry  # pyright: ignore[reportAssignmentType]
        y_hat = None
        for _ in range(cycles_per_data):
            carry, logits = model(carry, x)
            y_hat = torch.argmax(logits, dim=-1)
        total_correct += torch.all(y_hat == y, dim=-1).sum().item()
        total_samples += y.shape[0]
    if total_samples == 0:
        raise RuntimeError("evaluation loader was empty")
    return total_samples, total_correct / total_samples


def read_completed_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["train_band"]), int(row["L_cycles"]), str(row["test_band"]))].append(
            float(row["exact_match_accuracy"])
        )

    summaries: dict[tuple[str, str, int], dict[str, Any]] = {}
    for (model, train_band, l_cycles, test_band), values in grouped.items():
        summary = summaries.setdefault(
            (model, train_band, l_cycles),
            {"model": model, "train_band": train_band, "L_cycles": l_cycles},
        )
        summary[f"{test_band}_exact_match_mean"] = float(np.mean(values))
        summary[f"{test_band}_exact_match_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary["num_seeds"] = len(values)
    return sorted(summaries.values(), key=lambda item: (item["model"], item["train_band"], item["L_cycles"]))


def format_markdown_table(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "| model | train | L | easy exact match (mean +/- std) | hard exact match (mean +/- std) |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in summaries:
        easy = f"{row.get('easy_exact_match_mean', float('nan')):.4f} +/- {row.get('easy_exact_match_std', float('nan')):.4f}"
        hard = f"{row.get('hard_exact_match_mean', float('nan')):.4f} +/- {row.get('hard_exact_match_std', float('nan')):.4f}"
        lines.append(f"| {row['model']} | {row['train_band']} | {row['L_cycles']} | {easy} | {hard} |")
    return "\n".join(lines) + "\n"


def write_summaries(output_dir: Path, rows: list[dict[str, Any]]) -> str:
    summaries = summarize_rows(rows)
    summary_fields = [
        "model", "train_band", "L_cycles", "num_seeds",
        "easy_exact_match_mean", "easy_exact_match_std",
        "hard_exact_match_mean", "hard_exact_match_std",
    ]
    write_csv(output_dir / "test_time_scaling_summary.csv", summaries, summary_fields)
    table = format_markdown_table(summaries)
    (output_dir / "test_time_scaling_summary.md").write_text(table)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate k=55 Sudoku checkpoints at larger test-time recursion budgets.")
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/k55_test_time_scaling"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--conditions", nargs="+", choices=CONDITION_NAMES, default=list(CONDITION_NAMES),
        help="Subset of k=55 runs to evaluate; use one condition per GPU worker.",
    )
    parser.add_argument("--l-values", type=parse_l_values, default=L_VALUES)
    parser.add_argument("--batch-size", type=int, help="Override the checkpoint evaluation batch size")
    parser.add_argument("--wandb-entity", default="classicalqy-peking-university")
    parser.add_argument(
        "--selection-metric", default="eval/hard_exact_match",
        help="W&B metric used to choose each seed's best epoch (default: eval/hard_exact_match)",
    )
    parser.add_argument("--resume", action="store_true", help="Skip metric rows already present in the per-seed CSV")
    parser.add_argument(
        "--summarize-only", action="store_true",
        help="Merge --input-csv files and write the final seed-aggregated table without using a GPU or W&B.",
    )
    parser.add_argument(
        "--input-csv", action="append", type=Path,
        help="Per-seed CSV to include with --summarize-only; repeat once per worker output.",
    )
    args = parser.parse_args()

    if args.summarize_only:
        if not args.input_csv:
            parser.error("--summarize-only requires at least one --input-csv")
        rows = [row for path in args.input_csv for row in read_completed_rows(path)]
        if not rows:
            raise ValueError("the supplied per-seed CSV files contain no results")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        table = write_summaries(args.output_dir, rows)
        print("\n" + table)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("test-time scaling evaluation requires a CUDA GPU")
    if any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be unique non-negative integers")
    if len(set(args.conditions)) != len(args.conditions):
        parser.error("--conditions must not contain duplicates")

    runs = discover_runs(args.checkpoint_root, args.seeds, args.conditions)
    checkpoints, selections = select_best_checkpoints(runs, args.wandb_entity, args.selection_metric)
    data_name, data_kwargs, eval_sets = validate_shared_evaluation(checkpoints)

    batch_size = args.batch_size or int(checkpoints[0].config["local_batch_size"])
    loaders = create_eval_loaders(data_name, data_kwargs, eval_sets, batch_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "selected_checkpoints.csv",
        selections,
        [
            "condition", "model", "train_band", "seed", "selection_metric", "selection_exact_match",
            "wandb_step", "selected_epoch", "checkpoint", "wandb_run_id", "wandb_run_name",
        ],
    )
    per_seed_path = args.output_dir / "test_time_scaling_per_seed.csv"
    fields = [
        "condition", "model", "train_band", "seed", "L_cycles", "test_band",
        "total_samples", "exact_match_accuracy", "checkpoint",
    ]
    rows = read_completed_rows(per_seed_path) if args.resume else []
    selected_paths = {
        (checkpoint.condition, checkpoint.seed): checkpoint.path.resolve()
        for checkpoint in checkpoints
    }
    rows = [
        row for row in rows
        if selected_paths.get((row["condition"], int(row["seed"]))) == Path(row["checkpoint"]).resolve()
    ]
    if args.resume:
        write_csv(per_seed_path, rows, fields)
    completed = {
        (row["condition"], int(row["seed"]), int(row["L_cycles"]), row["test_band"])
        for row in rows
    }

    for checkpoint in checkpoints:
        for l_cycles in args.l_values:
            required = {
                (checkpoint.condition, checkpoint.seed, l_cycles, band)
                for band in ("easy", "hard")
            }
            if args.resume and required <= completed:
                print(f"Skipping completed {checkpoint.condition} seed={checkpoint.seed} L={l_cycles}")
                continue

            arch_options = architecture_at_l(checkpoint.config, checkpoint.model, l_cycles)
            model_cls = load_module(f"arch.{arch_options.pop('name')}")
            with torch.device("cuda"):
                model = model_cls(arch_options | {"vocab_size": 10, "seq_len": 82, "is_causal": False})
                state_dict = torch.load(checkpoint.path, map_location="cuda", weights_only=True)
                model.load_state_dict(state_dict, assign=True)
                model.eval()

            for test_band, loader in loaders.items():
                key = (checkpoint.condition, checkpoint.seed, l_cycles, test_band)
                if args.resume and key in completed:
                    continue
                total_samples, accuracy = evaluate_model(
                    model,
                    loader,
                    int(checkpoint.config["cycles_per_data"]),
                    f"{checkpoint.condition} seed={checkpoint.seed} L={l_cycles} {test_band}",
                )
                row = {
                    "condition": checkpoint.condition,
                    "model": checkpoint.model,
                    "train_band": checkpoint.train_band,
                    "seed": checkpoint.seed,
                    "L_cycles": l_cycles,
                    "test_band": test_band,
                    "total_samples": total_samples,
                    "exact_match_accuracy": accuracy,
                    "checkpoint": str(checkpoint.path),
                }
                rows.append(row)
                completed.add(key)
                write_csv(per_seed_path, rows, fields)
                print(f"{checkpoint.condition} seed={checkpoint.seed} L={l_cycles} {test_band}: {accuracy:.4f}")

            del model
            torch.cuda.empty_cache()

    table = write_summaries(args.output_dir, rows)
    print("\n" + table)


if __name__ == "__main__":
    main()

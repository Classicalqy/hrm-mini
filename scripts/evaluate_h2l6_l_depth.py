#!/usr/bin/env python
"""Evaluate L-depth inference curves for the three trained H2L6 readouts.

This script loads the selected best checkpoint for every H2L6-H/L/HL seed and
evaluates it at multiple inference-time L-cycle counts while preserving the native
H=2 and cycles_per_data=16 schedule.  Independent workers may shard the nine
(condition, seed) runs across GPUs; a separate merge invocation writes seed-summary
CSVs and figures.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import sys
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from arch.layers import Carry
from scripts.analyze_long_rollout_msd import RunDirectory, build_model, data_kwargs, load_config, load_module


CONDITIONS = ("H2L6_h", "H2L6_l", "H2L6_hl")
READOUT_BY_CONDITION = {"H2L6_h": "h", "H2L6_l": "l", "H2L6_hl": "hl"}
DEFAULT_L_VALUES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
FIELDS = [
    "condition", "readout", "seed", "checkpoint", "best_epoch", "eval_l",
    "test_exact_match", "cell_accuracy", "examples",
]
SUMMARY_FIELDS = [
    "condition", "readout", "eval_l", "seeds", "exact_mean", "exact_seed_sd",
    "cell_mean", "cell_seed_sd",
]


def parse_int_list(value: str) -> tuple[int, ...]:
    try:
        values = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers.") from error
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("At least one positive integer is required.")
    return values


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def atomic_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def absolute_checkpoint(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def selected_runs(best_path: Path, seeds: tuple[int, ...]) -> list[tuple[RunDirectory, dict[str, str], Path]]:
    rows = read_csv(best_path)
    selected = {(row.get("condition", ""), int(row.get("seed", -1))): row for row in rows}
    missing = [f"{condition}/seed_{seed}" for condition in CONDITIONS for seed in seeds if (condition, seed) not in selected]
    if missing:
        raise FileNotFoundError(f"Missing selected best checkpoints in {best_path}:\n" + "\n".join(missing))
    result = []
    for condition in CONDITIONS:
        for seed in seeds:
            row = selected[(condition, seed)]
            checkpoint = absolute_checkpoint(row["checkpoint"])
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Selected checkpoint is missing: {checkpoint}")
            config = load_config(checkpoint.parent)
            arch = config.arch.__pydantic_extra__ or {}
            if config.arch.name != "hrm@HRM" or arch.get("H_cycles") != 2 or arch.get("L_cycles") != 6:
                raise ValueError(f"{checkpoint} is not a standard H2L6 checkpoint.")
            if arch.get("readout") != READOUT_BY_CONDITION[condition]:
                raise ValueError(f"Readout mismatch for {checkpoint}.")
            result.append((RunDirectory("hrm", condition, seed, checkpoint.parent, config, 6, READOUT_BY_CONDITION[condition]), row, checkpoint))
    return result


@torch.inference_mode()
def evaluate_l_depth(
    model: torch.nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    *,
    eval_l: int,
    cycles_per_data: int,
    max_examples: int | None,
) -> tuple[float, float, int]:
    model.H_cycles, model.L_cycles = 2, eval_l  # type: ignore[attr-defined]
    exact = correct_cells = cells = examples = 0
    remaining = max_examples
    for x, y in loader:
        if remaining is not None and remaining <= 0:
            break
        if remaining is not None and x.shape[0] > remaining:
            x, y = x[:remaining], y[:remaining]
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        carry: Carry = model.initial_carry  # type: ignore[attr-defined]
        logits = None
        for _ in range(cycles_per_data):
            carry, logits = model(carry, x)
        assert logits is not None
        prediction = torch.argmax(logits, dim=-1)
        exact += torch.all(prediction == y, dim=-1).sum().item()
        correct_cells += (prediction[:, 1:] == y[:, 1:]).sum().item()
        examples += y.shape[0]
        cells += y[:, 1:].numel()
        if remaining is not None:
            remaining -= y.shape[0]
    if examples == 0:
        raise RuntimeError("Evaluation loader produced no examples.")
    return exact / examples, correct_cells / cells, examples


def row_key(row: dict[str, str] | dict[str, object]) -> tuple[str, int, int]:
    return str(row["condition"]), int(row["seed"]), int(row["eval_l"])


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["condition"], row["readout"], int(row["eval_l"])), []).append(row)
    result = []
    for (condition, readout, eval_l), group in sorted(grouped.items()):
        exact = np.asarray([float(row["test_exact_match"]) for row in group])
        cell = np.asarray([float(row["cell_accuracy"]) for row in group])
        result.append({
            "condition": condition, "readout": readout, "eval_l": eval_l, "seeds": len(group),
            "exact_mean": float(exact.mean()), "exact_seed_sd": float(exact.std(ddof=1)) if len(exact) > 1 else 0.0,
            "cell_mean": float(cell.mean()), "cell_seed_sd": float(cell.std(ddof=1)) if len(cell) > 1 else 0.0,
        })
    return result


def plot_curves(output_dir: Path, summary_rows: list[dict[str, object]], per_seed_rows: list[dict[str, str]]) -> None:
    figure, axes = plt.subplots(1, len(CONDITIONS), figsize=(5.2 * len(CONDITIONS), 4), sharey=True)
    for axis, condition in zip(np.ravel(axes), CONDITIONS):
        rows = sorted([row for row in summary_rows if row["condition"] == condition], key=lambda row: int(row["eval_l"]))
        x = np.asarray([int(row["eval_l"]) for row in rows])
        mean = 100 * np.asarray([float(row["exact_mean"]) for row in rows])
        spread = 100 * np.asarray([float(row["exact_seed_sd"]) for row in rows])
        for seed in (1, 2, 3):
            seed_rows = sorted([row for row in per_seed_rows if row["condition"] == condition and int(row["seed"]) == seed], key=lambda row: int(row["eval_l"]))
            if seed_rows:
                axis.plot([int(row["eval_l"]) for row in seed_rows], [100 * float(row["test_exact_match"]) for row in seed_rows],
                          color="tab:blue", alpha=.22, linewidth=1)
        axis.plot(x, mean, color="tab:blue", marker="o", linewidth=2, label="mean")
        axis.fill_between(x, mean - spread, mean + spread, color="tab:blue", alpha=.18, label="± seed SD")
        axis.axvline(6, color="black", linestyle="--", linewidth=1, alpha=.7, label="train L=6")
        axis.set_xscale("log", base=2)
        axis.set_title(condition)
        axis.set_xlabel("Inference L cycles")
        axis.grid(True, which="both", alpha=.25)
    axes[0].set_ylabel("test_hard exact match (%)")
    axes[-1].legend(fontsize=8)
    figure.suptitle("H2L6 inference-time L-depth ablation")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"l_depth_accuracy.{suffix}", dpi=200)
    plt.close(figure)


def write_readme(output_dir: Path, l_values: tuple[int, ...]) -> None:
    (output_dir / "README.md").write_text(f"""# H2L6 L-depth inference curves

The selected best H2L6-H, H2L6-L, and H2L6-HL checkpoints for seeds 1/2/3 are
evaluated on unaugmented `test_hard` at inference L values `{','.join(map(str, l_values))}`.
H remains fixed at 2 and `cycles_per_data=16` remains native. `l_depth_per_seed.csv`
contains all raw checkpoint results; `l_depth_seed_summary.csv` is the training-seed
mean and SD, not a puzzle bootstrap confidence interval.
""")


def merge_workers(source_dirs: list[Path], output_dir: Path, seeds: tuple[int, ...], l_values: tuple[int, ...]) -> None:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, int, int]] = set()
    for source in source_dirs:
        worker_rows = read_csv(source / "l_depth_per_seed.csv")
        if not worker_rows:
            raise FileNotFoundError(f"Missing l_depth_per_seed.csv in {source}")
        for row in worker_rows:
            key = row_key(row)
            if key in seen:
                raise ValueError(f"Duplicate worker result for {key}")
            seen.add(key)
            rows.append(row)
    expected = {(condition, seed, eval_l) for condition in CONDITIONS for seed in seeds for eval_l in l_values}
    actual = {row_key(row) for row in rows}
    if actual != expected:
        raise ValueError(f"Worker merge incomplete: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: (row["condition"], int(row["seed"]), int(row["eval_l"])))
    summary_rows = summarize(rows)
    atomic_csv(output_dir / "l_depth_per_seed.csv", rows, FIELDS)
    atomic_csv(output_dir / "l_depth_seed_summary.csv", summary_rows, SUMMARY_FIELDS)
    plot_curves(output_dir, summary_rows, rows)
    write_readme(output_dir, l_values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best-checkpoints", type=Path, default=Path("results/core_five_long_rollout/final_absolute/best_checkpoints.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/h2l6_l_depth_inference"))
    parser.add_argument("--seeds", type=parse_int_list, default=(1, 2, 3))
    parser.add_argument("--eval-l-values", type=parse_int_list, default=DEFAULT_L_VALUES)
    parser.add_argument("--samples", type=int, default=0, help="0 evaluates all effective test_hard examples; positive values make a smoke run.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--merge-from", type=Path, nargs="+", metavar="WORKER_DIR")
    args = parser.parse_args()
    if args.samples < 0:
        parser.error("--samples must be non-negative.")
    if args.num_shards < 1:
        parser.error("--num-shards must be positive.")
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    if args.merge_from:
        if args.shard_index is not None or args.num_shards != 1:
            parser.error("--merge-from cannot be combined with shard options.")
        merge_workers(args.merge_from, output_dir, args.seeds, args.eval_l_values)
        return
    if args.num_shards != 1 and args.shard_index is None:
        parser.error("--num-shards requires --shard-index.")
    if args.shard_index is not None and not 0 <= args.shard_index < args.num_shards:
        parser.error(f"--shard-index must be in [0, {args.num_shards - 1}].")
    best_path = args.best_checkpoints if args.best_checkpoints.is_absolute() else PROJECT_ROOT / args.best_checkpoints
    runs = selected_runs(best_path, args.seeds)
    if args.shard_index is not None:
        runs = runs[args.shard_index::args.num_shards]
    if not runs:
        parser.error("This worker received no (condition, seed) runs.")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "l_depth_per_seed.csv"
    rows = read_csv(result_path)
    completed = {row_key(row) for row in rows}
    first_config = runs[0][0].config
    create_dataloader = load_module(f"dataset.{first_config.data.name}@create_dataloader")
    loader, metadata = create_dataloader("test_hard", first_config.local_batch_size, rank=0, world_size=1, **data_kwargs(first_config))
    total = sum(1 for run, _, _ in runs for l_value in args.eval_l_values if (run.condition, run.seed, l_value) not in completed)
    progress = tqdm(total=total, desc="H2L6 L-depth evaluation", unit="model×L")
    for run, selected, checkpoint in runs:
        pending = [l_value for l_value in args.eval_l_values if (run.condition, run.seed, l_value) not in completed]
        if not pending:
            continue
        model = build_model(run, checkpoint, metadata, torch.device(args.device))
        for eval_l in pending:
            progress.set_postfix_str(f"{run.condition}/seed_{run.seed}, L={eval_l}")
            exact, cell, examples = evaluate_l_depth(model, loader, torch.device(args.device), eval_l=eval_l,
                                                     cycles_per_data=run.config.cycles_per_data,
                                                     max_examples=args.samples or None)
            row = {
                "condition": run.condition, "readout": run.readout, "seed": run.seed,
                "checkpoint": str(checkpoint), "best_epoch": int(selected["epoch"]), "eval_l": eval_l,
                "test_exact_match": exact, "cell_accuracy": cell, "examples": examples,
            }
            rows = [old for old in rows if row_key(old) != row_key(row)] + [row]
            atomic_csv(result_path, rows, FIELDS)
            progress.update(1)
        del model
        gc.collect()
        if torch.device(args.device).type == "cuda":
            torch.cuda.empty_cache()
    progress.close()
    (output_dir / "worker_metadata.json").write_text(json.dumps({
        "seeds": args.seeds, "eval_l_values": args.eval_l_values, "shard_index": args.shard_index,
        "num_shards": args.num_shards, "samples": args.samples or "all effective test_hard examples",
        "best_checkpoints": str(best_path),
    }, indent=2) + "\n")
    expected = {(condition, seed, l_value) for condition in CONDITIONS for seed in args.seeds for l_value in args.eval_l_values}
    if {row_key(row) for row in rows} == expected:
        summary_rows = summarize(rows)
        atomic_csv(output_dir / "l_depth_seed_summary.csv", summary_rows, SUMMARY_FIELDS)
        plot_curves(output_dir, summary_rows, rows)
        write_readme(output_dir, args.eval_l_values)


if __name__ == "__main__":
    main()

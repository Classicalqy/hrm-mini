#!/usr/bin/env python
"""Causal H-state interventions for the trained H2L6 readout controls.

The evaluator runs the native 16 outer-cycle schedule (32 H boundaries) on
H2L6-H/L/HL checkpoints selected by the core-five evaluator.  It compares normal
inference with two interventions applied at a chosen H-block index:

* ``stale_h``: every subsequent L block receives the H state from one block ago;
* ``freeze_h``: keep the H state fixed and continue updating L normally.

The H2L6-L checkpoint is the cleanest readout control for interpreting an H-to-L
effect, since its logits are decoded from L rather than directly from H.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Literal

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
MODES = ("normal", "stale_h", "freeze_h")
DEFAULT_STARTS = (0, 1, 2, 4, 8, 16, 31)
TRAJECTORY_FIELDS = [
    "condition", "readout", "seed", "checkpoint", "best_epoch", "mode", "start_h_block", "h_block",
    "exact_match", "cell_accuracy", "mean_hamming_cells", "mean_target_margin", "examples",
]
FINAL_FIELDS = [field for field in TRAJECTORY_FIELDS if field != "h_block"]
SUMMARY_FIELDS = [
    "condition", "readout", "mode", "start_h_block", "metric", "seeds", "normal_mean",
    "intervention_mean", "delta_mean", "delta_seed_sd",
]


def parse_int_list(value: str) -> tuple[int, ...]:
    try:
        result = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated integers.") from error
    if not result:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    return result


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


def checkpoint_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def selected_runs(best_path: Path, seeds: tuple[int, ...]) -> list[tuple[RunDirectory, dict[str, str], Path]]:
    rows = read_csv(best_path)
    selected: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        condition, seed = row.get("condition", ""), int(row.get("seed", -1))
        if condition in CONDITIONS and seed in seeds:
            selected[(condition, seed)] = row
    missing = [f"{condition}/seed_{seed}" for condition in CONDITIONS for seed in seeds if (condition, seed) not in selected]
    if missing:
        raise FileNotFoundError(
            f"{best_path} lacks selected checkpoints for:\n" + "\n".join(missing) +
            "\nRun scripts/analyze_long_rollout_msd.py --profile core-five first, or pass --best-checkpoints."
        )
    runs: list[tuple[RunDirectory, dict[str, str], Path]] = []
    for condition in CONDITIONS:
        for seed in seeds:
            row = selected[(condition, seed)]
            checkpoint = checkpoint_path(row["checkpoint"])
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Selected checkpoint is missing: {checkpoint}")
            config = load_config(checkpoint.parent)
            arch = config.arch.__pydantic_extra__ or {}
            if config.arch.name != "hrm@HRM" or arch.get("H_cycles") != 2 or arch.get("L_cycles") != 6:
                raise ValueError(f"{checkpoint} is not an H2L6 standard-HRM checkpoint.")
            if arch.get("readout") != READOUT_BY_CONDITION[condition]:
                raise ValueError(f"Readout mismatch for {checkpoint}: expected {READOUT_BY_CONDITION[condition]!r}.")
            run = RunDirectory("hrm", condition, seed, checkpoint.parent, config, 6, READOUT_BY_CONDITION[condition])
            runs.append((run, row, checkpoint))
    return runs


@torch.inference_mode()
def advance_h_block(
    model: torch.nn.Module,
    carry: Carry,
    input_ids: torch.Tensor,
    *,
    h_override: torch.Tensor | None = None,
    frozen_h: torch.Tensor | None = None,
) -> tuple[Carry, torch.Tensor]:
    """Run one native H boundary (six L updates, then H), with optional H intervention."""
    z_h = carry["z_H"] if h_override is None else h_override
    z_l = carry["z_L"]
    x = model.embed(input_ids)  # type: ignore[attr-defined]
    for _ in range(model.L_cycles):  # type: ignore[attr-defined]
        z_l = model.L_level(z_l + z_h + x)  # type: ignore[attr-defined]
    z_h_next = frozen_h if frozen_h is not None else model.H_level(z_h + z_l)  # type: ignore[attr-defined]
    logits = model.readout_logits(z_h_next, z_l)  # type: ignore[attr-defined]
    return {"z_H": z_h_next.detach(), "z_L": z_l.detach()}, logits


def metric_accumulators(blocks: int) -> list[dict[str, float]]:
    return [dict(exact=0.0, cells=0.0, hamming=0.0, margin=0.0, examples=0.0, cell_total=0.0) for _ in range(blocks)]


def add_metrics(accumulator: dict[str, float], logits: torch.Tensor, targets: torch.Tensor) -> None:
    prediction = torch.argmax(logits, dim=-1)
    cells_prediction, cells_targets = prediction[:, 1:], targets[:, 1:]
    examples, cells = cells_targets.shape
    accumulator["exact"] += torch.all(prediction == targets, dim=-1).sum().item()
    accumulator["cells"] += (cells_prediction == cells_targets).sum().item()
    accumulator["hamming"] += (cells_prediction != cells_targets).sum(dim=-1).sum().item()
    target_logits = logits[:, 1:].gather(-1, cells_targets.unsqueeze(-1)).squeeze(-1)
    masked = logits[:, 1:].clone()
    masked.scatter_(-1, cells_targets.unsqueeze(-1), float("-inf"))
    accumulator["margin"] += (target_logits - masked.max(dim=-1).values).sum().item()
    accumulator["examples"] += examples
    accumulator["cell_total"] += examples * cells


def scenario_key(row: dict[str, str] | dict[str, object]) -> tuple[str, int, str, int]:
    return str(row["condition"]), int(row["seed"]), str(row["mode"]), int(row["start_h_block"])


@torch.inference_mode()
def evaluate_scenario(
    model: torch.nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    *,
    mode: Literal["normal", "stale_h", "freeze_h"],
    start_h_block: int,
    h_blocks: int,
    max_examples: int | None,
    progress: tqdm[Any] | None,
) -> list[dict[str, float]]:
    totals = metric_accumulators(h_blocks)
    remaining = max_examples
    for x, y in loader:
        if remaining is not None and remaining <= 0:
            break
        if remaining is not None and x.shape[0] > remaining:
            x, y = x[:remaining], y[:remaining]
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        carry: Carry = model.initial_carry  # type: ignore[attr-defined]
        history = [carry["z_H"]]
        frozen_h: torch.Tensor | None = None
        for block in range(h_blocks):
            h_override = None
            if mode == "stale_h" and block >= start_h_block and block > 0:
                h_override = history[-2]
            if mode == "freeze_h" and block >= start_h_block:
                if frozen_h is None:
                    frozen_h = carry["z_H"]
            carry, logits = advance_h_block(model, carry, x, h_override=h_override, frozen_h=frozen_h)
            history.append(carry["z_H"])
            add_metrics(totals[block], logits, y)
        if remaining is not None:
            remaining -= x.shape[0]
        if progress is not None:
            progress.update(1)
    if any(total["examples"] == 0 for total in totals):
        raise RuntimeError("Evaluation loader produced no examples.")
    return totals


def rows_for_scenario(
    run: RunDirectory,
    selected: dict[str, str],
    checkpoint: Path,
    mode: str,
    start: int,
    totals: list[dict[str, float]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = []
    for block, total in enumerate(totals, start=1):
        row: dict[str, object] = {
            "condition": run.condition, "readout": run.readout, "seed": run.seed,
            "checkpoint": str(checkpoint), "best_epoch": int(selected["epoch"]),
            "mode": mode, "start_h_block": start, "h_block": block,
            "exact_match": total["exact"] / total["examples"],
            "cell_accuracy": total["cells"] / total["cell_total"],
            "mean_hamming_cells": total["hamming"] / total["examples"],
            "mean_target_margin": total["margin"] / total["cell_total"],
            "examples": int(total["examples"]),
        }
        rows.append(row)
    final = dict(rows[-1])
    final.pop("h_block")
    return rows, final


def summarize(final_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    normal = {(row["condition"], int(row["seed"])): row for row in final_rows if row["mode"] == "normal"}
    grouped: dict[tuple[str, str, str, int], list[tuple[dict[str, str], dict[str, str]]]] = {}
    for row in final_rows:
        if row["mode"] == "normal":
            continue
        base = normal.get((row["condition"], int(row["seed"])))
        if base is not None:
            grouped.setdefault((row["condition"], row["readout"], row["mode"], int(row["start_h_block"])), []).append((base, row))
    rows: list[dict[str, object]] = []
    for (condition, readout, mode, start), pairs in sorted(grouped.items()):
        for metric in ("exact_match", "cell_accuracy", "mean_hamming_cells", "mean_target_margin"):
            normal_values = np.asarray([float(base[metric]) for base, _ in pairs])
            intervention_values = np.asarray([float(row[metric]) for _, row in pairs])
            delta = intervention_values - normal_values
            rows.append({
                "condition": condition, "readout": readout, "mode": mode, "start_h_block": start,
                "metric": metric, "seeds": len(pairs), "normal_mean": float(normal_values.mean()),
                "intervention_mean": float(intervention_values.mean()), "delta_mean": float(delta.mean()),
                "delta_seed_sd": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
            })
    return rows


def plot_final_exact(output_dir: Path, summary_rows: list[dict[str, object]]) -> None:
    figure, axes = plt.subplots(1, len(CONDITIONS), figsize=(5.2 * len(CONDITIONS), 4), sharey=True)
    for axis, condition in zip(np.ravel(axes), CONDITIONS):
        for mode, color in (("stale_h", "tab:orange"), ("freeze_h", "tab:red")):
            rows = sorted(
                [row for row in summary_rows if row["condition"] == condition and row["mode"] == mode and row["metric"] == "exact_match"],
                key=lambda row: int(row["start_h_block"]),
            )
            if not rows:
                continue
            starts = np.asarray([int(row["start_h_block"]) for row in rows])
            delta = 100 * np.asarray([float(row["delta_mean"]) for row in rows])
            spread = 100 * np.asarray([float(row["delta_seed_sd"]) for row in rows])
            axis.errorbar(starts, delta, yerr=spread, marker="o", capsize=3, color=color, label=mode)
        axis.axhline(0, color="black", linewidth=.8)
        axis.set_title(condition)
        axis.set_xlabel("Intervention start H-block")
        axis.grid(alpha=.25)
    axes[0].set_ylabel("Final exact-match change (pp; mean ± seed SD)")
    axes[-1].legend()
    figure.suptitle("Native-horizon H-state interventions")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"final_exact_delta.{suffix}", dpi=200)
    plt.close(figure)


def write_readme(path: Path, starts: tuple[int, ...]) -> None:
    path.write_text(f"""# H2L6 H-state intervention evaluation

This directory evaluates the selected H2L6-H, H2L6-L, and H2L6-HL checkpoints over
their native 16 outer cycles (32 H boundaries).  Results use unaugmented `test_hard`.

Interventions start at H-block indices `{','.join(map(str, starts))}` (zero-indexed):

* `normal`: no state intervention;
* `stale_h`: each subsequent L block consumes the H state from one H block earlier;
* `freeze_h`: H is held fixed at its value immediately before the start block while L
  continues to update.

`intervention_trajectory.csv` records every H boundary. `intervention_final.csv`
contains final native-rollout outcomes. `intervention_seed_summary.csv` reports
per-seed paired changes from the corresponding normal run. In H-readout models,
freeze-H has a direct readout confound; H2L6-L is the cleanest test of whether H
affects subsequent L-mediated prediction.
""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best-checkpoints", type=Path, default=Path("results/core_five_long_rollout/final_absolute/best_checkpoints.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/h2l6_h_interventions"))
    parser.add_argument("--seeds", type=parse_int_list, default=(1, 2, 3))
    parser.add_argument("--starts", type=parse_int_list, default=DEFAULT_STARTS, help="Zero-indexed H blocks at which stale/freeze begins.")
    parser.add_argument("--samples", type=int, default=0, help="0 evaluates all effective test_hard examples; positive values make a smoke run.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.samples < 0:
        parser.error("--samples must be non-negative.")
    device = torch.device(args.device)
    best_path = args.best_checkpoints if args.best_checkpoints.is_absolute() else PROJECT_ROOT / args.best_checkpoints
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    runs = selected_runs(best_path, args.seeds)
    h_blocks = runs[0][0].config.cycles_per_data * 2
    if any(start < 0 or start >= h_blocks for start in args.starts):
        parser.error(f"--starts must lie in [0, {h_blocks - 1}] for the native rollout.")
    scenarios = [("normal", -1)] + [(mode, start) for mode in ("stale_h", "freeze_h") for start in args.starts]
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path, final_path = output_dir / "intervention_trajectory.csv", output_dir / "intervention_final.csv"
    trajectory_rows = read_csv(trajectory_path)
    final_rows = read_csv(final_path)
    completed = {scenario_key(row) for row in final_rows}

    first_config = runs[0][0].config
    create_dataloader = load_module(f"dataset.{first_config.data.name}@create_dataloader")
    loader, metadata = create_dataloader("test_hard", first_config.local_batch_size, rank=0, world_size=1, **data_kwargs(first_config))
    effective_examples = args.samples if args.samples else None
    batch_count = math.ceil((effective_examples or 19968) / first_config.local_batch_size)
    progress = tqdm(total=sum(1 for run, _, _ in runs for mode, start in scenarios if (run.condition, run.seed, mode, start) not in completed) * batch_count,
                    desc="H2L6 interventions", unit="batch")
    for run, selected, checkpoint in runs:
        pending = [(mode, start) for mode, start in scenarios if (run.condition, run.seed, mode, start) not in completed]
        if not pending:
            continue
        model = build_model(run, checkpoint, metadata, device)
        for mode, start in pending:
            progress.set_postfix_str(f"{run.condition}/seed_{run.seed}, {mode}@{start}")
            totals = evaluate_scenario(model, loader, device, mode=mode, start_h_block=start, h_blocks=h_blocks,
                                       max_examples=effective_examples, progress=progress)
            scenario_trajectory, scenario_final = rows_for_scenario(run, selected, checkpoint, mode, start, totals)
            key = scenario_key(scenario_final)
            trajectory_rows = [row for row in trajectory_rows if scenario_key(row) != key] + scenario_trajectory
            final_rows = [row for row in final_rows if scenario_key(row) != key] + [scenario_final]
            atomic_csv(trajectory_path, trajectory_rows, TRAJECTORY_FIELDS)
            atomic_csv(final_path, final_rows, FINAL_FIELDS)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    progress.close()
    summary_rows = summarize(final_rows)
    atomic_csv(output_dir / "intervention_seed_summary.csv", summary_rows, SUMMARY_FIELDS)
    plot_final_exact(output_dir, summary_rows)
    (output_dir / "analysis_metadata.json").write_text(json.dumps({
        "conditions": CONDITIONS, "seeds": args.seeds, "starts": args.starts,
        "native_outer_cycles": runs[0][0].config.cycles_per_data, "native_h_boundaries": h_blocks,
        "samples": "all effective test_hard examples" if args.samples == 0 else args.samples,
        "best_checkpoints": str(best_path),
    }, indent=2) + "\n")
    write_readme(output_dir / "README.md", args.starts)


if __name__ == "__main__":
    main()

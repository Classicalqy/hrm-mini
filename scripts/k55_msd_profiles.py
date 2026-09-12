"""K55 checkpoint discovery, best-epoch selection, and L-depth MSD profiles."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from scripts.analyze_long_rollout_msd import RunDirectory, epoch_checkpoints, load_config, load_module
from scripts.core_five_long_rollout import (
    fixed_random_samples,
    native_accuracy_with_count,
    read_csv,
)
from scripts.core_five_l_depth_long_rollout import (
    L_DEPTH_COLORS,
    METADATA_FIELDS,
    RATIO_FIELDS,
    SEED_FIELDS,
    SELECTION_FIELDS,
    SUMMARY_FIELDS,
    SweepUnit,
    atomic_csv,
    cluster_bounds,
    collect_unit,
    curve,
    lags_for_boundaries,
    plot_positive,
    ratio_rows,
    rollout_spec,
    summarize,
)
from scripts.analyze_long_rollout_msd import build_model, data_kwargs


K55_DIRECTORY = re.compile(r"(?P<difficulty>easy|hard)_k55_(?P<model>hrm_h2l1|hrm|trm|rt)$")
SEED_DIRECTORY = re.compile(r"seed_(?P<seed>\d+)$")
DIFFICULTIES = ("easy", "hard")
MODEL_ORDER = ("hrm", "trm", "hrm_h2l1", "rt")


def condition_name(difficulty: str, model: str) -> str:
    return f"{difficulty}_k55_{model}"


def discover_k55_runs(root: Path, seeds: tuple[int, ...]) -> list[RunDirectory]:
    """Discover the exact easy/hard K55 directory layout requested by the profile."""
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint root does not exist: {root}")
    found: dict[tuple[str, int], RunDirectory] = {}
    for condition_dir in root.iterdir():
        match = K55_DIRECTORY.fullmatch(condition_dir.name)
        if match is None:
            continue
        model_name = match["model"]
        for seed_dir in condition_dir.iterdir():
            seed_match = SEED_DIRECTORY.fullmatch(seed_dir.name) if seed_dir.is_dir() else None
            if seed_match is None or not epoch_checkpoints(seed_dir):
                continue
            seed = int(seed_match["seed"])
            if seed not in seeds:
                continue
            config = load_config(seed_dir)
            arch = config.arch.__pydantic_extra__ or {}
            expected_arch = {
                "hrm": "hrm@HRM",
                "hrm_h2l1": "hrm@HRM",
                "trm": "trm@TRM",
                "rt": "rt@RecurrentTransformer",
            }[model_name]
            if config.arch.name != expected_arch:
                raise ValueError(f"{seed_dir}: expected {expected_arch}, got {config.arch.name}.")
            kind = "rt" if model_name == "rt" else ("trm" if model_name == "trm" else "hrm")
            if kind == "rt":
                l_cycles, readout = None, "rt"
            else:
                l_cycles = int(arch.get("L_cycles", 0))
                readout = "h"
                expected_l = 1 if model_name == "hrm_h2l1" else 6
                if int(arch.get("H_cycles", 0)) != 2 or l_cycles < 1:
                    raise ValueError(f"{seed_dir}: expected H_cycles=2 and positive L_cycles, got {arch}.")
                if l_cycles != expected_l:
                    raise ValueError(f"{seed_dir}: {model_name} requires L_cycles={expected_l}, got {l_cycles}.")
                if kind == "hrm" and arch.get("readout", "h") != "h":
                    raise ValueError(f"{seed_dir}: K55 HRM profiles require the H readout.")
            found[(condition_dir.name, seed)] = RunDirectory(
                kind=kind, condition=condition_dir.name, seed=seed, directory=seed_dir,
                config=config, l_cycles=l_cycles, readout=readout,
            )
    expected = [(condition_name(d, m), seed) for d in DIFFICULTIES for m in MODEL_ORDER for seed in seeds]
    missing = [f"{condition}/seed_{seed}" for condition, seed in expected if (condition, seed) not in found]
    if missing:
        raise FileNotFoundError("Missing K55 checkpoint runs:\n" + "\n".join(missing))
    return [found[key] for key in expected]


def difficulty_of(condition: str) -> str:
    return condition.split("_", 1)[0]


def make_loader(run: RunDirectory, split: str):
    create = load_module(f"dataset.{run.config.data.name}@create_dataloader")
    return create(split, run.config.local_batch_size, rank=0, world_size=1, **data_kwargs(run.config))


def select_k55_best(runs: list[RunDirectory], output_dir: Path, device: torch.device, split: str) -> list[dict[str, object]]:
    """Select one native-schedule epoch independently for every condition and seed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "best_checkpoints.csv"
    expected = {(run.condition, run.seed) for run in runs}
    existing = {
        (row["condition"], int(row["seed"])): dict(row)
        for row in read_csv(destination)
        if (row["condition"], int(row["seed"])) in expected and Path(row["checkpoint"]).is_file()
    }
    pending = [run for run in runs if (run.condition, run.seed) not in existing]
    progress = tqdm(total=sum(len(epoch_checkpoints(run.directory)) for run in pending), desc="Select K55 best checkpoints", unit="epoch")
    loaders: dict[str, tuple[Any, dict[str, Any]]] = {}
    for run in pending:
        difficulty = difficulty_of(run.condition)
        if difficulty not in loaders:
            loaders[difficulty] = make_loader(run, split)
        loader, metadata = loaders[difficulty]
        best: dict[str, object] | None = None
        for epoch, checkpoint in epoch_checkpoints(run.directory):
            progress.set_postfix_str(f"{run.condition}/seed_{run.seed}, epoch={epoch}")
            model = build_model(run, checkpoint, metadata, device)
            exact, cell, examples = native_accuracy_with_count(model, run, loader, device)
            candidate = {
                "kind": run.kind, "condition": run.condition, "readout": run.readout,
                "train_l": "" if run.l_cycles is None else run.l_cycles, "seed": run.seed,
                "epoch": epoch, "checkpoint": str(checkpoint), "test_exact_match": exact,
                "cell_accuracy": cell, "evaluated_examples": examples,
                "selection_completed": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            if best is None or exact > float(best["test_exact_match"]):
                best = candidate
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            progress.update(1)
        assert best is not None
        existing[(run.condition, run.seed)] = best
        atomic_csv(destination, [existing[key] for key in sorted(existing)], SELECTION_FIELDS)
    progress.close()
    return [existing[(run.condition, run.seed)] for run in runs]


def k55_units(runs: list[RunDirectory], l_values: tuple[int, ...]) -> list[SweepUnit]:
    units = []
    for run in runs:
        model_name = K55_DIRECTORY.fullmatch(run.condition)["model"]  # type: ignore[index]
        values = l_values if model_name in ("hrm", "trm") else (1,)
        units.extend(SweepUnit(run, value) for value in values)
    return units


def _comparison_figures(output_dir: Path, args: Any) -> None:
    metadata = read_csv(output_dir / "rollout_metadata.csv")
    seed_summary = read_csv(output_dir / "msd_seed_cluster_bootstrap.csv")
    for difficulty in DIFFICULTIES:
        figure, axes = plt.subplots(2, 4, figsize=(15, 6), sharex=True, sharey=True)
        for row, model_name in enumerate(("hrm", "trm")):
            condition = condition_name(difficulty, model_name)
            for segment, axis in enumerate(axes[row]):
                for eval_l in args.l_depth_values:
                    lags, mean, _low, _high, seeds = curve(metadata, condition, eval_l, "hl_concat", segment)
                    low, high = cluster_bounds(seed_summary, condition, eval_l, "hl_concat", segment, lags)
                    plot_positive(axis, lags, mean, low, high, L_DEPTH_COLORS[eval_l], f"L={eval_l}", seeds)
                for control_model, style, color in (("hrm_h2l1", "--", "#222222"), ("rt", ":", "#777777")):
                    control = condition_name(difficulty, control_model)
                    state = "rt" if control_model == "rt" else "hl_concat"
                    lags, mean, _low, _high, seeds = curve(metadata, control, 1, state, segment)
                    for seed_curve in seeds:
                        axis.plot(lags, seed_curve, style, color=color, linewidth=.6, alpha=.25)
                    axis.plot(lags, mean, style, color=color, linewidth=1.2, label=control_model)
                axis.set_xscale("log", base=2); axis.set_yscale("log", base=2); axis.grid(alpha=.2, which="both")
                axis.set_title(f"{model_name.upper()}, segment {segment + 1}")
                if segment == 0: axis.set_ylabel("joint / RT per-coordinate MSD")
                if row == 1: axis.set_xlabel("lag (underlying updates)")
        axes[0, -1].legend(fontsize=7, ncol=2)
        figure.suptitle(f"K55 {difficulty}: HRM/TRM inference-L sweep")
        figure.tight_layout()
        for suffix in ("png", "pdf"):
            figure.savefig(output_dir / f"k55_{difficulty}_l_depth_comparison.{suffix}", dpi=200)
        plt.close(figure)


def finalize_k55_l_depth(args: Any) -> None:
    puzzle_rows, seed_rows = summarize(args.output_dir, args.bootstrap_replicates, args.sample_seed)
    atomic_csv(args.output_dir / "msd_puzzle_bootstrap.csv", puzzle_rows, SUMMARY_FIELDS)
    atomic_csv(args.output_dir / "msd_seed_cluster_bootstrap.csv", seed_rows, SEED_FIELDS)
    ratios = ratio_rows(read_csv(args.output_dir / "rollout_metadata.csv"))
    atomic_csv(args.output_dir / "h_over_l_ratio.csv", ratios, RATIO_FIELDS)
    _comparison_figures(args.output_dir, args)
    (args.output_dir / "analysis_metadata.json").write_text(json.dumps({
        "profile": "k55-l-depth", "selection_split": args.k55_split,
        "eval_l_values": args.l_depth_values, "seeds": args.core_seeds,
        "samples": args.samples, "sample_seed": args.sample_seed,
        "bootstrap_replicates": args.bootstrap_replicates, "max_l_updates": 4096,
        "lag_points": args.lag_points,
    }, indent=2) + "\n")
    (args.output_dir / "README.md").write_text(f"""# K55 inference-L MSD

The native best epoch was selected independently for every model/seed using
`{args.k55_split}` exact match (earlier epoch wins a tie). HRM and TRM are swept
over inference L values `{','.join(map(str, args.l_depth_values))}`. H2L1-H and
RT remain fixed controls. Every rollout uses 4,096 underlying recurrent updates;
MSD is the per-puzzle, time-averaged, full-state per-coordinate squared
displacement. `h_plus_l` includes the H/L displacement cross term, whereas
`hl_concat` is the normalized concatenated state and equals `(MSD_H+MSD_L)/2`.
""")


def main_k55_core(args: Any) -> None:
    if args.merge_from:
        runs = discover_k55_runs(args.checkpoints_root, args.core_seeds)
        expected = {(run.condition, run.seed) for run in runs}
        rows: dict[tuple[str, int], dict[str, str]] = {}
        for worker in args.merge_from:
            for row in read_csv(Path(worker) / "best_checkpoints.csv"):
                key = (row["condition"], int(row["seed"]))
                if key in rows:
                    raise ValueError(f"K55 selection workers overlap at {key}.")
                rows[key] = row
        if set(rows) != expected:
            raise ValueError(f"Incomplete K55 selection merge: missing={sorted(expected - set(rows))}, extra={sorted(set(rows) - expected)}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_csv(args.output_dir / "best_checkpoints.csv", [rows[key] for key in sorted(rows)], SELECTION_FIELDS)
        return
    runs = discover_k55_runs(args.checkpoints_root, args.core_seeds)
    if args.shard_index is not None:
        runs = runs[args.shard_index::args.num_shards]
    if not runs:
        raise ValueError("This K55 selection shard received no runs.")
    select_k55_best(runs, args.output_dir, args.device, args.k55_split)


def merge_k55_l_depth(args: Any, all_units: list[SweepUnit]) -> None:
    import filecmp

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata: list[dict[str, str]] = []
    selections: dict[tuple[str, int], dict[str, str]] = {}
    for difficulty in DIFFICULTIES:
        manifests = [Path(worker) / f"sample_manifest_{difficulty}.sha256" for worker in args.merge_from]
        manifests = [path for path in manifests if path.is_file()]
        hashes = [path.read_text().strip() for path in manifests]
        if not hashes or len(set(hashes)) != 1:
            raise ValueError(f"K55 {difficulty} worker sample manifests are missing or differ.")
        source_dir = manifests[0].parent
        shutil.copy2(source_dir / f"sample_manifest_{difficulty}.csv", args.output_dir / f"sample_manifest_{difficulty}.csv")
        shutil.copy2(manifests[0], args.output_dir / f"sample_manifest_{difficulty}.sha256")
    for worker in args.merge_from:
        worker_path = Path(worker)
        metadata.extend(read_csv(worker_path / "rollout_metadata.csv"))
        for row in read_csv(worker_path / "best_checkpoints.csv"):
            selections[(row["condition"], int(row["seed"]))] = row
        for source in (worker_path / "per_puzzle_msd").glob("*.npz"):
            target = args.output_dir / "per_puzzle_msd" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if not filecmp.cmp(source, target, shallow=False):
                    raise ValueError(f"Conflicting K55 result file: {source.name}")
            else:
                shutil.copy2(source, target)
    expected = {unit.key for unit in all_units}
    actual = {(row["condition"], int(row["seed"]), int(row["eval_l"])) for row in metadata}
    if actual != expected or len(metadata) != len(actual):
        raise ValueError(f"Incomplete/overlapping K55 rollout merge: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    for row in metadata:
        row["per_puzzle_file"] = str(args.output_dir / "per_puzzle_msd" / Path(row["per_puzzle_file"]).name)
    atomic_csv(args.output_dir / "rollout_metadata.csv", metadata, METADATA_FIELDS)
    atomic_csv(args.output_dir / "best_checkpoints.csv", list(selections.values()), SELECTION_FIELDS)
    finalize_k55_l_depth(args)


def main_k55_l_depth(args: Any) -> None:
    runs = discover_k55_runs(args.checkpoints_root, args.core_seeds)
    all_units = k55_units(runs, args.l_depth_values)
    if args.merge_from:
        merge_k55_l_depth(args, all_units)
        return
    selected_rows = read_csv(args.reference_best_checkpoints)
    selected = {(row["condition"], int(row["seed"])): row for row in selected_rows}
    missing = sorted({(unit.run.condition, unit.run.seed) for unit in all_units} - set(selected))
    if missing:
        raise FileNotFoundError(f"Reference best-checkpoint CSV is incomplete: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.output_dir / "rollout_metadata.csv")
    completed = {(row["condition"], int(row["seed"]), int(row["eval_l"])) for row in rows}
    assigned = all_units[args.shard_index::args.num_shards] if args.shard_index is not None else all_units
    if not assigned:
        raise ValueError("This K55 L-depth shard received no units.")
    for difficulty in DIFFICULTIES:
        group = [unit for unit in assigned if difficulty_of(unit.run.condition) == difficulty]
        if not group:
            continue
        loader, metadata = make_loader(group[0].run, args.k55_split)
        fixed_x, indices = fixed_random_samples(loader, args.samples, args.sample_seed)
        digest = hashlib.sha256(",".join(map(str, indices)).encode()).hexdigest()
        manifest_rows = [{"sample_position": i, "stream_index": int(v), "sample_seed": args.sample_seed} for i, v in enumerate(indices)]
        atomic_csv(args.output_dir / f"sample_manifest_{difficulty}.csv", manifest_rows,
                   ["sample_position", "stream_index", "sample_seed"])
        (args.output_dir / f"sample_manifest_{difficulty}.sha256").write_text(digest + "\n")
        chunks = math.ceil(len(fixed_x) / args.rollout_batch_size)
        pending = [unit for unit in group if unit.key not in completed]
        specs = {unit.key: rollout_spec(unit, args) for unit in pending}
        progress = tqdm(total=sum(len(lags_for_boundaries(bounds, args.lag_points)) * chunks
                                  for _updates, bounds, _scheme in specs.values()),
                        desc=f"K55 {difficulty} L-depth rollouts", unit="lag-batch")
        for unit in pending:
            updates, boundaries, scheme = specs[unit.key]
            lags = lags_for_boundaries(boundaries, args.lag_points)
            progress.set_postfix_str(unit.label)
            row = collect_unit(unit, selected[(unit.run.condition, unit.run.seed)], fixed_x, metadata, args,
                               digest, lags, updates, boundaries, scheme, progress)
            rows = [old for old in rows if (old["condition"], int(old["seed"]), int(old["eval_l"])) != unit.key] + [row]
            atomic_csv(args.output_dir / "rollout_metadata.csv", rows, METADATA_FIELDS)
        progress.close()
    atomic_csv(args.output_dir / "best_checkpoints.csv", list(selected.values()), SELECTION_FIELDS)
    if args.num_shards == 1:
        finalize_k55_l_depth(args)


__all__ = ["discover_k55_runs", "k55_units", "main_k55_core", "main_k55_l_depth", "merge_k55_l_depth"]

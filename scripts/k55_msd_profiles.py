"""K55 checkpoint discovery, best-epoch selection, and L-depth MSD profiles."""

from __future__ import annotations

from functools import partial
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import time
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from scripts.analyze_long_rollout_msd import RunDirectory, epoch_checkpoints, load_config
from scripts.core_five_long_rollout import (
    fixed_random_samples,
    native_accuracy_with_count,
    read_csv,
)
from scripts.core_five_l_depth_long_rollout import (
    METADATA_FIELDS,
    RATIO_FIELDS,
    SEED_FIELDS,
    SELECTION_FIELDS,
    SUMMARY_FIELDS,
    SweepUnit,
    atomic_csv,
    collect_unit,
    curve,
    lags_for_boundaries,
    ratio_rows,
    rollout_spec,
    summarize,
)
from scripts.analyze_long_rollout_msd import build_model

if TYPE_CHECKING:
    from train import TrainConfig


K55_DIRECTORY = re.compile(r"(?P<difficulty>easy|hard)_k55_(?P<model>hrm_h2l1|hrm|trm|rt)$")
SEED_DIRECTORY = re.compile(r"seed_(?P<seed>\d+)$")
DIFFICULTIES = ("easy", "hard")
MODEL_ORDER = ("hrm", "trm", "hrm_h2l1", "rt")


def condition_name(difficulty: str, model: str) -> str:
    return f"{difficulty}_k55_{model}"


def inferred_k55_config(condition: str) -> TrainConfig:
    """Reconstruct the immutable training config when only a state dict was downloaded."""
    from train import TrainConfig

    match = K55_DIRECTORY.fullmatch(condition)
    if match is None:
        raise ValueError(f"Not a K55 condition: {condition}")
    difficulty, model_name = match["difficulty"], match["model"]
    if model_name == "rt":
        arch = {
            "name": "rt@RecurrentTransformer", "num_layers": 4, "hidden_size": 512,
            "intermediate_size": 2048, "head_dim": 64, "norm_eps": 1e-6,
            "rope_theta": 10000.0, "cycles": 7, "bptt": True, "forward_dtype": "bfloat16",
        }
    else:
        arch = {
            "name": "trm@TRM" if model_name == "trm" else "hrm@HRM",
            "num_layers": 2, "hidden_size": 512, "intermediate_size": 2048,
            "head_dim": 64, "norm_eps": 1e-6, "rope_theta": 10000.0,
            "H_cycles": 2, "L_cycles": 1 if model_name == "hrm_h2l1" else 6,
            "bptt": True, "forward_dtype": "bfloat16",
        }
    data = {
        "name": "sudoku", "dataset_name": "./downloaded-datasets/sudoku-extreme-full",
        "eval_dataset_name": "./downloaded-datasets/sudoku-extreme-full",
        "num_base_puzzles": 1000, "repeat": 200, "augment": True,
        "eval_sets": {
            "easy": {"split": "test", "eval_blank_max": 55},
            "hard": {"split": "test", "eval_blank_min": 56},
        },
        "eval_num_base_puzzles": 10000, "eval_seed": 42,
    }
    data["blank_max" if difficulty == "easy" else "blank_min"] = 55 if difficulty == "easy" else 56
    return TrainConfig(**{
        "arch": arch, "data": data, "seeds": [1, 2, 3], "cycles_per_data": 16,
        "epochs": 20, "local_batch_size": 96, "lr": 1e-4, "lr_warmup_steps": 2000,
        "lr_min_ratio": 1.0, "beta1": .9, "beta2": .95, "weight_decay": 1.0, "ema": .999,
    })


def load_k55_config(seed_dir: Path, condition: str) -> TrainConfig:
    """Load plain metadata, bypassing legacy cyclic OmegaConf YAML objects."""
    metadata_path = seed_dir / "model_config.json"
    if metadata_path.is_file():
        contents = metadata_path.read_text()
        if "!!python/object:" not in contents and "omegaconf." not in contents:
            return load_config(seed_dir)
    return inferred_k55_config(condition)


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
            config = load_k55_config(seed_dir, condition_dir.name)
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


def make_loader(run: RunDirectory, requested_band: str):
    """Build the K55 evaluation loader using the config's named band definition."""
    from dataset.sudoku import collate_fn
    from datasets import Features, Value, load_dataset

    band = difficulty_of(run.condition) if requested_band == "matched" else requested_band
    if band not in ("easy", "hard"):
        raise ValueError(f"Unsupported K55 evaluation band: {requested_band}")
    data = dict(run.config.data.__pydantic_extra__ or {})
    eval_sets = data.get("eval_sets") or {
        "easy": {"split": "test", "eval_blank_max": 55},
        "hard": {"split": "test", "eval_blank_min": 56},
    }
    options = dict(eval_sets[band])
    split = options.pop("split", "test")
    source = data.get("eval_dataset_name") or data["dataset_name"]
    features = Features({
        "source": Value("string"),
        "question": Value("string"),
        "answer": Value("string"),
        "rating": Value("int64"),
    })
    try:
        dataset = load_dataset(source, split=split, features=features)
    except Exception as error:
        cause = error.__cause__ or error
        raise RuntimeError(
            f"Failed to load K55 dataset {source!r}, split={split!r}: "
            f"{type(cause).__name__}: {cause}"
        ) from error
    lower = options.get("eval_blank_min")
    upper = options.get("eval_blank_max")
    if lower is not None or upper is not None:
        minimum = int(lower) if lower is not None else 0
        maximum = int(upper) if upper is not None else 81
        dataset = dataset.filter(lambda question: minimum <= question.count(".") <= maximum, input_columns="question")
    requested = data.get("eval_num_base_puzzles")
    if requested is not None:
        requested = int(requested)
        if len(dataset) < requested:
            raise ValueError(f"K55 {band} requested {requested} evaluation puzzles but only {len(dataset)} are available.")
        dataset = dataset.shuffle(seed=int(data.get("eval_seed", 42))).select(range(requested))
    sampler = DistributedSampler(dataset, rank=0, num_replicas=1, shuffle=False, drop_last=True, seed=42)
    loader = DataLoader(
        dataset, batch_size=run.config.local_batch_size,
        collate_fn=partial(collate_fn, augment=False), sampler=sampler,
        drop_last=True, pin_memory=True, num_workers=1, persistent_workers=True, prefetch_factor=2,
    )
    return loader, {"vocab_size": 10, "seq_len": 82, "is_causal": False}


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
    segment_colors = plt.get_cmap("tab10")

    def seed1_curve(condition: str, eval_l: int, state: str, segment: int) -> tuple[np.ndarray, np.ndarray]:
        seed_ids = sorted(
            int(item["seed"]) for item in metadata
            if item["condition"] == condition and int(item["eval_l"]) == eval_l
        )
        if 1 not in seed_ids:
            raise ValueError(f"Seed 1 is missing for {condition}/L{eval_l}.")
        lags, _mean, _low, _high, seed_curves = curve(metadata, condition, eval_l, state, segment)
        return lags, seed_curves[seed_ids.index(1)]

    def configure_msd_axis(axis: Any, condition: str, eval_l: int, title: str) -> None:
        for segment in range(4):
            color = segment_colors(segment)
            for state, style, marker, width in (("h", "-", "o", 2.0), ("l", "--", "^", 1.8)):
                lags, values = seed1_curve(condition, eval_l, state, segment)
                plot_lags = lags
                defined = values > 0
                if state == "h":
                    # Preserve the original phase-averaged H curve after the
                    # first complete H period, but hide its initial sub-cycle line.
                    defined &= lags >= eval_l
                plotted = np.where(defined, values, np.nan)
                axis.plot(plot_lags, plotted, style, color=color, marker=marker, markersize=3.2,
                          markerfacecolor=color if state == "h" else "white",
                          markeredgewidth=.8, linewidth=width, zorder=3 if state == "h" else 2)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=2)
        axis.grid(alpha=.2, which="both")
        axis.set_title(title)

    def configure_rt_axis(axis: Any) -> None:
        condition = condition_name("easy", "rt")
        for segment in range(4):
            lags, values = seed1_curve(condition, 1, "rt", segment)
            axis.plot(lags, np.where(values > 0, values, np.nan), ":", color=segment_colors(segment),
                      marker="s", markersize=3, linewidth=1.9)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=2)
        axis.grid(alpha=.2, which="both")
        axis.set_title("RT control")

    def ratio_curve(condition: str, eval_l: int, segment: int) -> tuple[np.ndarray, np.ndarray]:
        h_lags, h_values = seed1_curve(condition, eval_l, "h", segment)
        l_lags, l_values = seed1_curve(condition, eval_l, "l", segment)
        if not np.array_equal(h_lags, l_lags):
            raise AssertionError(f"H/L lag mismatch for {condition}/L{eval_l}/segment{segment + 1}.")
        ratio = np.full_like(h_values, np.nan, dtype=np.float64)
        defined = np.logical_and(h_values > 0, l_values > 0)
        defined &= h_lags >= eval_l
        ratio[defined] = h_values[defined] / l_values[defined]
        return h_lags, ratio

    def configure_ratio_axis(
        axis: Any, condition: str, eval_l: int, model_label: str, title: str,
    ) -> None:
        h2l1_condition = condition_name("easy", "hrm_h2l1")
        for segment in range(4):
            h_lags, ratio = ratio_curve(condition, eval_l, segment)
            axis.plot(
                h_lags, ratio, color=segment_colors(segment), marker="D", markersize=3,
                linewidth=2.0, label=model_label if segment == 0 else None,
            )
            control_lags, control_ratio = ratio_curve(h2l1_condition, 1, segment)
            axis.plot(
                control_lags, control_ratio, "--", color=segment_colors(segment), marker="x",
                markersize=3.2, linewidth=1.25, alpha=.72,
                label="H2L1" if segment == 0 else None,
            )
        axis.axhline(1, color="0.35", linestyle=":", linewidth=1)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=2)
        axis.grid(alpha=.2, which="both")
        axis.set_title(title)
        axis.legend(loc="best", fontsize=7, frameon=False)

    def log_log_slope(condition: str, eval_l: int, state: str, segment: int) -> float:
        lags, values = seed1_curve(condition, eval_l, state, segment)
        defined = np.logical_and(values > 0, lags >= eval_l)
        if np.count_nonzero(defined) < 2:
            return math.nan
        return float(np.polyfit(np.log2(lags[defined]), np.log2(values[defined]), 1)[0])

    hierarchical = (
        # Deliberately distinct from the blue/orange/green/red segment palette
        # used by the first six panels: color means model in the slope panels.
        ("HRM", condition_name("easy", "hrm"), 6, "#6f00ff"),
        ("TRM", condition_name("easy", "trm"), 6, "#008b8b"),
        ("H2L1", condition_name("easy", "hrm_h2l1"), 1, "#d89000"),
    )
    model_markers = {"HRM": "o", "TRM": "s", "H2L1": "D"}

    def slope_series(condition: str, eval_l: int, state: str) -> np.ndarray:
        return np.asarray([
            log_log_slope(condition, eval_l, state, segment) for segment in range(4)
        ])

    def configure_slope_axis(axis: Any) -> None:
        segments = np.arange(1, 5)
        for label, condition, eval_l, color in hierarchical:
            axis.plot(
                segments, slope_series(condition, eval_l, "h"), "-o", color=color,
                linewidth=1.8, markersize=4,
            )
            axis.plot(
                segments, slope_series(condition, eval_l, "l"), "--^", color=color,
                markerfacecolor="white", linewidth=1.5, markersize=4,
            )
        rt_condition = condition_name("easy", "rt")
        axis.plot(
            segments, slope_series(rt_condition, 1, "rt"), ":s", color="#222222",
            linewidth=1.8, markersize=4,
        )
        axis.axhline(0, color="0.5", linestyle=":", linewidth=1)
        axis.set_xticks(segments, [f"S{segment}" for segment in segments])
        axis.set_ylabel("β")
        axis.set_xlabel("rollout segment")
        axis.set_title("Absolute slopes β")
        axis.grid(alpha=.2)
        model_handles = [
            Line2D([0], [0], color=color, linewidth=2, label=label)
            for label, _, _, color in hierarchical
        ]
        model_handles.append(
            Line2D([0], [0], color="#222222", linestyle=":", linewidth=2, label="RT")
        )
        axis.legend(handles=model_handles, ncol=2, fontsize=7, frameon=False,
                    columnspacing=.9, handlelength=2.2)

    def configure_slope_ratio_axis(axis: Any) -> None:
        segments = np.arange(1, 5)
        for label, condition, eval_l, color in hierarchical:
            h_slopes = slope_series(condition, eval_l, "h")
            l_slopes = slope_series(condition, eval_l, "l")
            ratios = np.divide(
                h_slopes, l_slopes, out=np.full_like(h_slopes, np.nan),
                where=np.abs(l_slopes) > np.finfo(float).eps,
            )
            axis.plot(
                segments, ratios, color=color, marker=model_markers[label], linewidth=2.2,
                markersize=4.5, label=label,
            )
        axis.axhline(1, color="0.35", linestyle=":", linewidth=1)
        axis.set_xticks(segments, [f"S{segment}" for segment in segments])
        axis.set_ylabel("β(H) / β(L)")
        axis.set_xlabel("rollout segment")
        axis.set_title("Slope ratio β(H)/β(L)")
        axis.grid(alpha=.2)
        axis.legend(fontsize=7, frameon=False)

    stored_boundaries = {
        tuple(json.loads(row["segment_boundaries_l_updates"])) for row in metadata
    }
    if len(stored_boundaries) != 1:
        raise ValueError(f"K55 plotting requires one shared section layout, got {stored_boundaries}.")
    boundary_points = next(iter(stored_boundaries))
    boundaries = tuple(zip(boundary_points[:-1], boundary_points[1:]))
    segment_handles = [
        Line2D([0], [0], color=segment_colors(segment), linewidth=2,
               label=f"S{segment + 1}  {start}–{end}")
        for segment, (start, end) in enumerate(boundaries)
    ]
    state_handles = [
        Line2D([0], [0], color="#555555", linestyle="-", marker="o", linewidth=2, label="H"),
        Line2D([0], [0], color="#555555", linestyle="--", marker="^", markerfacecolor="white",
               linewidth=1.8, label="L"),
        Line2D([0], [0], color="#222222", linestyle=":", marker="s", linewidth=1.9, label="RT state"),
    ]
    for primary_eval_l in (6, 8, 16, 32):
        hierarchical = (
            ("HRM", condition_name("easy", "hrm"), primary_eval_l, "#6f00ff"),
            ("TRM", condition_name("easy", "trm"), primary_eval_l, "#008b8b"),
            ("H2L1", condition_name("easy", "hrm_h2l1"), 1, "#d89000"),
        )
        figure, axes = plt.subplots(2, 4, figsize=(16, 8))
        configure_msd_axis(
            axes[0, 0], condition_name("easy", "hrm"), primary_eval_l,
            f"HRM (L={primary_eval_l})",
        )
        configure_msd_axis(
            axes[0, 1], condition_name("easy", "trm"), primary_eval_l,
            f"TRM (L={primary_eval_l})",
        )
        configure_msd_axis(axes[0, 2], condition_name("easy", "hrm_h2l1"), 1, "H2L1 control")
        configure_rt_axis(axes[0, 3])
        configure_ratio_axis(
            axes[1, 0], condition_name("easy", "hrm"), primary_eval_l,
            "HRM", "MSD ratio: HRM vs H2L1",
        )
        configure_ratio_axis(
            axes[1, 1], condition_name("easy", "trm"), primary_eval_l,
            "TRM", "MSD ratio: TRM vs H2L1",
        )
        ratio_axes = list(axes[1, :2])
        ratio_limits = (
            min(axis.get_ylim()[0] for axis in ratio_axes),
            max(axis.get_ylim()[1] for axis in ratio_axes),
        )
        for axis in ratio_axes:
            axis.set_ylim(ratio_limits)
        configure_slope_axis(axes[1, 2])
        configure_slope_ratio_axis(axes[1, 3])

        msd_axes = [axes[0, column] for column in range(4)]
        common_limits = (
            min(axis.get_ylim()[0] for axis in msd_axes),
            max(axis.get_ylim()[1] for axis in msd_axes),
        )
        for axis in msd_axes:
            axis.set_ylim(common_limits)
        axes[0, 0].set_ylabel("per-coordinate MSD")
        axes[1, 0].set_ylabel("MSD(H) / MSD(L)")
        for axis in axes[1, :2]:
            axis.set_xlabel("lag (underlying L updates)")

        segment_legend = figure.legend(
            handles=segment_handles, loc="upper center", bbox_to_anchor=(.36, .945),
            ncol=4, fontsize=8,
        )
        figure.add_artist(segment_legend)
        figure.legend(
            handles=state_handles, loc="upper center", bbox_to_anchor=(.76, .945), ncol=3, fontsize=8,
        )
        figure.suptitle("Deterministic long-rollout dynamics", y=.995)
        figure.tight_layout(rect=(0, 0, 1, .91))
        stem = "k55_easy_h_l_models_overview_seed1"
        if primary_eval_l != 6:
            stem += f"_L{primary_eval_l}"
        for suffix in ("png", "pdf"):
            figure.savefig(output_dir / f"{stem}.{suffix}", dpi=200)
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
    if args.plots_only:
        if not (args.output_dir / "rollout_metadata.csv").is_file():
            raise FileNotFoundError(f"--plots-only requires an existing merged K55 result in {args.output_dir}")
        _comparison_figures(args.output_dir, args)
        return
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
    unit_lookup = {unit.key: unit for unit in all_units}
    completed: set[tuple[str, int, int]] = set()
    for row in rows:
        key = (row["condition"], int(row["seed"]), int(row["eval_l"]))
        unit = unit_lookup.get(key)
        if unit is None:
            continue
        updates, expected_boundaries, expected_scheme = rollout_spec(unit, args)
        recorded_boundaries = np.asarray((), dtype=np.int64)
        try:
            recorded_boundaries = np.asarray(
                json.loads(row["segment_boundaries_l_updates"]), dtype=np.int64,
            )
            cache_path = args.output_dir / "per_puzzle_msd" / unit_filename(unit)
            with np.load(cache_path, allow_pickle=False) as cached:
                cache_matches = (
                    "analysis_scheme" in cached.files
                    and str(cached["analysis_scheme"].item()) == expected_scheme
                    and np.array_equal(cached["segment_boundaries_l_updates"], expected_boundaries)
                )
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            cache_matches = False
        if (
            int(row["actual_l_updates"]) == updates
            and np.array_equal(recorded_boundaries, expected_boundaries)
            and cache_matches
        ):
            completed.add(key)
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

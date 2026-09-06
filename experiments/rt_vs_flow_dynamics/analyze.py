from __future__ import annotations

import csv
from itertools import combinations
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .trace import detect_fixed_points, trajectory_metrics


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _bootstrap_seed_puzzle(
    values: dict[int, np.ndarray], samples: int, generator: np.random.Generator
) -> tuple[float, float]:
    seeds = np.asarray(sorted(values))
    draws = np.empty(samples, dtype=np.float64)
    for draw in range(samples):
        selected = generator.choice(seeds, size=len(seeds), replace=True)
        seed_means = []
        for seed in selected:
            data = values[int(seed)]
            seed_means.append(generator.choice(data, size=len(data), replace=True).mean())
        draws[draw] = np.mean(seed_means)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def analyze_traces(config: dict[str, Any]) -> Path:
    root = Path(config["experiment"]["output_root"])
    trace_dir = root / "traces"
    output_dir = root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_config = config["trace"]
    records: list[tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]] = []
    per_run: list[dict[str, object]] = []
    per_puzzle: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    fingerprints: dict[tuple[str, int], str] = {}
    for path in sorted(trace_dir.glob("*.npz")):
        with np.load(path) as data:
            arrays = {key: data[key] for key in data.files}
        metadata = json.loads(path.with_suffix(".json").read_text())
        metrics = trajectory_metrics(arrays)
        onset = detect_fixed_points(
            arrays, int(trace_config["fixed_window"]), float(trace_config["state_eps"]),
            float(trace_config["logit_eps"]),
        )
        condition, seed = str(metadata["condition"]), int(metadata["seed"])
        # Flow discretization checks share a checkpoint but are distinct analysis
        # conditions.  Keeping NFE in the key prevents 56/112/224 traces from
        # silently overwriting one another during aggregation.
        analysis_condition = (
            f"{condition}_nfe_{metadata['steps']}" if metadata.get("model_kind") == "flow" else condition
        )
        final_exact = arrays["exact_match"][:, -1].astype(np.float32)
        puzzle_metrics = metrics | {
            "final_exact_match": final_exact,
            "wrong_attractor": (onset >= 0).astype(np.float32),
            "settling_time": np.where(onset >= 0, onset, np.nan).astype(np.float32),
        }
        per_puzzle[(analysis_condition, seed)] = puzzle_metrics
        fingerprints[(analysis_condition, seed)] = str(metadata["dataset_fingerprint"])
        row: dict[str, object] = {
            "condition": analysis_condition, "seed": seed, "steps": int(metadata["steps"]),
            "samples": len(final_exact), "final_exact_match": float(final_exact.mean()),
            "wrong_attractor_rate": float((onset >= 0).mean()),
            "mean_wrong_settling_time": float(np.nanmean(puzzle_metrics["settling_time"])) if np.any(onset >= 0) else float("nan"),
        }
        row |= {key: float(value.mean()) for key, value in metrics.items()}
        for threshold in trace_config.get("sensitivity_state_eps", []):
            sensitivity = detect_fixed_points(
                arrays, int(trace_config["fixed_window"]), float(threshold), float(trace_config["logit_eps"])
            )
            row[f"wrong_rate_state_eps_{threshold:g}"] = float((sensitivity >= 0).mean())
        per_run.append(row)
        records.append((metadata, arrays, metrics, onset))
    if not records:
        raise FileNotFoundError(f"no trace NPZ files found under {trace_dir}")
    _write_csv(output_dir / "per_run_summary.csv", per_run)

    generator = np.random.default_rng(int(config["analysis"]["bootstrap_seed"]))
    bootstrap_samples = int(config["analysis"]["bootstrap_samples"])
    aggregate: list[dict[str, object]] = []
    metric_names = (
        "final_exact_match", "wrong_attractor", "tortuosity", "direction_autocorrelation",
        "backtrack_rate", "prediction_reversals", "effective_rank",
    )
    conditions = sorted({condition for condition, _ in per_puzzle})
    for condition in conditions:
        seeds = sorted(seed for candidate, seed in per_puzzle if candidate == condition)
        for metric in metric_names:
            values = {seed: per_puzzle[(condition, seed)][metric] for seed in seeds}
            seed_means = np.asarray([np.nanmean(values[seed]) for seed in seeds])
            low, high = _bootstrap_seed_puzzle(values, bootstrap_samples, generator)
            aggregate.append({
                "condition": condition, "metric": metric, "seed_count": len(seeds),
                "mean": float(seed_means.mean()),
                "sample_sd": float(seed_means.std(ddof=1)) if len(seeds) > 1 else float("nan"),
                "ci_low": low, "ci_high": high,
            })
    _write_csv(output_dir / "aggregate_summary.csv", aggregate)

    paired: list[dict[str, object]] = []
    for left, right in combinations(conditions, 2):
        common_seeds = sorted(
            set(seed for candidate, seed in per_puzzle if candidate == left)
            & set(seed for candidate, seed in per_puzzle if candidate == right)
        )
        for metric in metric_names:
            differences = []
            for seed in common_seeds:
                a, b = per_puzzle[(left, seed)][metric], per_puzzle[(right, seed)][metric]
                if len(a) == len(b) and fingerprints[(left, seed)] == fingerprints[(right, seed)]:
                    differences.append(float(np.nanmean(a - b)))
            if differences:
                paired.append({
                    "left": left, "right": right, "metric": metric, "seed_count": len(differences),
                    "mean_paired_difference": float(np.mean(differences)),
                    "sample_sd": float(np.std(differences, ddof=1)) if len(differences) > 1 else float("nan"),
                })
    _write_csv(output_dir / "paired_differences.csv", paired)
    _plot_behavior(records, output_dir)
    _plot_geometry(aggregate, output_dir)
    return output_dir


def _plot_behavior(
    records: list[tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]],
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for metadata, arrays, _, _ in records:
        label = f"{metadata['condition']}/s{metadata['seed']}"
        x = np.arange(arrays["exact_match"].shape[1])
        axes[0].plot(x, arrays["exact_match"].mean(axis=0), alpha=.75, label=label)
        axes[1].plot(x, arrays["constraint_violations"].mean(axis=0), alpha=.75)
        axes[2].semilogy(x[1:], np.clip(arrays["state_residual"][:, 1:].mean(axis=0), 1e-8, None), alpha=.75)
    for axis, title in zip(axes, ("Exact match", "Constraint violations", "Normalized state residual"), strict=True):
        axis.set_title(title); axis.set_xlabel("micro-step"); axis.grid(alpha=.25)
    axes[0].legend(fontsize=7)
    figure.tight_layout(); figure.savefig(output_dir / "behavior.png", dpi=180); plt.close(figure)


def _plot_geometry(rows: list[dict[str, object]], output_dir: Path) -> None:
    metrics = ("tortuosity", "backtrack_rate", "effective_rank", "wrong_attractor")
    conditions = sorted({str(row["condition"]) for row in rows})
    figure, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    for axis, metric in zip(axes, metrics, strict=True):
        selected = [row for condition in conditions for row in rows if row["condition"] == condition and row["metric"] == metric]
        means = [float(row["mean"]) for row in selected]
        lows = [mean - float(row["ci_low"]) for mean, row in zip(means, selected, strict=True)]
        highs = [float(row["ci_high"]) - mean for mean, row in zip(means, selected, strict=True)]
        axis.bar(range(len(selected)), means, yerr=np.asarray([lows, highs]), capsize=3)
        axis.set_xticks(range(len(selected)), [str(row["condition"]) for row in selected], rotation=45, ha="right")
        axis.set_title(metric); axis.grid(axis="y", alpha=.25)
    figure.tight_layout(); figure.savefig(output_dir / "geometry_summary.png", dpi=180); plt.close(figure)


def plot_toy_results(config: dict[str, Any]) -> Path:
    root = Path(config["experiment"]["output_root"]) / "toy"
    rows = list(csv.DictReader((root / "metrics.csv").open()))
    conditions = ("toy_rt", "flow_linear", "flow_curve")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for condition in conditions:
        selected = [row for row in rows if row["condition"] == condition]
        axes[0].plot([row["family"] for row in selected], [float(row["path_rmse"]) for row in selected], marker="o", label=condition)
        axes[1].plot([row["family"] for row in selected], [float(row["tortuosity"]) for row in selected], marker="o", label=condition)
    axes[0].set_ylabel("path RMSE"); axes[1].set_ylabel("tortuosity")
    for axis in axes:
        axis.tick_params(axis="x", rotation=35); axis.grid(alpha=.25)
    axes[0].legend(); figure.tight_layout(); figure.savefig(root / "toy_summary.png", dpi=180); plt.close(figure)
    return root / "toy_summary.png"

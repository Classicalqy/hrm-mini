"""Statistical long-rollout evaluation for the five HRM mechanism controls.

This module is invoked by ``analyze_long_rollout_msd.py --profile core-five``.
It deliberately keeps the older full-sweep workflow untouched while adding
per-puzzle MSD, bootstrap statistics, resumable checkpoint selection, and compact
comparison figures for the core five conditions.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from arch.layers import Carry
from scripts.analyze_long_rollout_msd import (
    RunDirectory,
    build_model,
    data_kwargs,
    discover_hrm_runs,
    discover_rt_runs,
    epoch_checkpoints,
    load_module,
)
from scripts.long_rollout_msd_utils import (
    STATE_NAMES,
    local_log_slope,
    rt_state_msd_per_puzzle,
    segment_boundaries,
    segment_for_pair,
    segment_lags,
    state_msd_per_puzzle,
)


CORE_CONDITIONS = ("H2L1_h", "H2L6_h", "H2L6_l", "H2L6_hl", "RT")
CORE_ORDER = {condition: index for index, condition in enumerate(CORE_CONDITIONS)}
SELECTION_FIELDS = [
    "kind", "condition", "readout", "train_l", "seed", "epoch", "checkpoint",
    "test_exact_match", "cell_accuracy", "evaluated_examples", "selection_completed",
]
ROLLOUT_FIELDS = [
    "kind", "condition", "readout", "train_l", "seed", "best_epoch", "checkpoint", "samples",
    "sample_seed", "sample_manifest_sha256", "requested_l_updates", "actual_l_updates", "total_blocks",
    "segment_boundaries_blocks", "lag_blocks", "state_shape", "per_puzzle_file",
]
PUZZLE_SUMMARY_FIELDS = [
    "kind", "condition", "readout", "train_l", "seed", "state", "segment", "lag_blocks", "lag_l_updates",
    "mean", "median", "ci95_low", "ci95_high", "puzzles", "origins",
]
SEED_SUMMARY_FIELDS = [
    "kind", "condition", "readout", "train_l", "state", "segment", "lag_blocks", "lag_l_updates",
    "seed_mean", "cluster_ci95_low", "cluster_ci95_high", "seeds", "puzzles_per_seed",
]
BETA_FIELDS = [
    "kind", "condition", "readout", "train_l", "seed", "state", "segment", "lag_blocks", "lag_l_updates",
    "beta_mean", "beta_median", "ci95_low", "ci95_high", "puzzles",
]
MOBILITY_FIELDS = [
    "condition", "readout", "seed", "segment", "lag_blocks", "lag_l_updates", "h_over_l",
    "interaction", "interaction_fraction_h_plus_l",
]


def atomic_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(sorted({int(item) for item in value.split(",") if item}))
    except ValueError as error:
        raise ValueError("Seeds must be comma-separated integers.") from error
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def core_runs(root: Path, seeds: tuple[int, ...]) -> list[RunDirectory]:
    discovered = discover_hrm_runs(root) + discover_rt_runs(root)
    by_key = {(run.condition, run.seed): run for run in discovered if run.condition in CORE_ORDER}
    missing = [f"{condition}/seed_{seed}" for condition in CORE_CONDITIONS for seed in seeds if (condition, seed) not in by_key]
    if missing:
        raise FileNotFoundError("Missing required core-five checkpoint directories:\n" + "\n".join(missing))
    return [by_key[(condition, seed)] for condition in CORE_CONDITIONS for seed in seeds]


@torch.inference_mode()
def native_accuracy_with_count(
    model: torch.nn.Module, run: RunDirectory, loader: Iterable[tuple[torch.Tensor, torch.Tensor]], device: torch.device,
) -> tuple[float, float, int]:
    if run.kind == "hrm":
        model.H_cycles = 2  # type: ignore[attr-defined]
        model.L_cycles = run.l_cycles  # type: ignore[attr-defined]
    exact = examples = correct_cells = cells = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        carry: Carry = model.initial_carry  # type: ignore[attr-defined]
        prediction = None
        for _ in range(run.config.cycles_per_data):
            carry, logits = model(carry, x)
            prediction = torch.argmax(logits, dim=-1)
        assert prediction is not None
        exact += torch.all(prediction == y, dim=-1).sum().item()
        examples += y.shape[0]
        correct_cells += (prediction[:, 1:] == y[:, 1:]).sum().item()
        cells += y[:, 1:].numel()
    if not examples:
        raise RuntimeError("test_hard produced no effective examples.")
    return exact / examples, correct_cells / cells, examples


def select_best_resumable(
    runs: list[RunDirectory], loader: Iterable[tuple[torch.Tensor, torch.Tensor]], metadata: dict[str, Any],
    device: torch.device, output_dir: Path,
) -> list[dict[str, object]]:
    destination = output_dir / "best_checkpoints.csv"
    existing = read_csv(destination)
    valid_existing: dict[tuple[str, int], dict[str, object]] = {}
    expected = {(run.condition, run.seed) for run in runs}
    for row in existing:
        key = (row["condition"], int(row["seed"]))
        if key in expected and Path(row["checkpoint"]).is_file():
            valid_existing[key] = dict(row)
    pending = [run for run in runs if (run.condition, run.seed) not in valid_existing]
    progress = tqdm(total=sum(len(epoch_checkpoints(run.directory)) for run in pending), desc="Select core best checkpoints", unit="epoch")
    for run in pending:
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
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            progress.update(1)
        assert best is not None
        valid_existing[(run.condition, run.seed)] = best
        ordered = [valid_existing[(run.condition, run.seed)] for run in runs if (run.condition, run.seed) in valid_existing]
        atomic_csv(destination, ordered, SELECTION_FIELDS)
    progress.close()
    return [valid_existing[(run.condition, run.seed)] for run in runs]


def fixed_random_samples(loader: Iterable[tuple[torch.Tensor, torch.Tensor]], samples: int, seed: int) -> tuple[torch.Tensor, np.ndarray]:
    batches = [x for x, _ in loader]
    stream = torch.cat(batches, dim=0)
    if samples > stream.shape[0]:
        raise ValueError(f"Requested {samples} puzzles but only {stream.shape[0]} effective test examples are available.")
    rng = np.random.default_rng(seed)
    stream_indices = np.sort(rng.choice(stream.shape[0], size=samples, replace=False)).astype(np.int64)
    return stream[torch.from_numpy(stream_indices)], stream_indices


def write_manifest(output_dir: Path, indices: np.ndarray, sample_seed: int) -> str:
    rows = [{"sample_position": index, "test_hard_stream_index": int(value), "sample_seed": sample_seed} for index, value in enumerate(indices)]
    atomic_csv(output_dir / "sample_manifest.csv", rows, ["sample_position", "test_hard_stream_index", "sample_seed"])
    digest = hashlib.sha256(",".join(str(value) for value in indices).encode()).hexdigest()
    (output_dir / "sample_manifest.sha256").write_text(digest + "\n")
    return digest


def _initial_hrm(model: torch.nn.Module, carry: Carry, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h, l = carry["z_H"], carry["z_L"]
    shape = (x.shape[0], x.shape[1], h.shape[-1])
    if h.ndim < 3:
        h = h.reshape(1, 1, -1).expand(shape)
    if l.ndim < 3:
        l = l.reshape(1, 1, -1).expand(shape)
    return h, l


@torch.inference_mode()
def _advance_hrm(model: torch.nn.Module, carry: Carry, x: torch.Tensor) -> tuple[Carry, tuple[torch.Tensor, torch.Tensor]]:
    state: tuple[torch.Tensor, torch.Tensor] | None = None
    def callback(event: str, h: torch.Tensor, l: torch.Tensor) -> None:
        nonlocal state
        if event == "h":
            state = (h, l)
    carry, _ = model.forward_with_trace(carry, x, callback, events=("h",))  # type: ignore[attr-defined]
    if state is None:
        raise RuntimeError("Missing H-boundary trace state.")
    return carry, state


@torch.inference_mode()
def _advance_rt(model: torch.nn.Module, carry: Carry, x: torch.Tensor) -> tuple[Carry, torch.Tensor]:
    state: torch.Tensor | None = None
    def callback(event: str, z: torch.Tensor) -> None:
        nonlocal state
        if event == "z":
            state = z
    carry, _ = model.forward_with_trace(carry, x, callback, events=("z",))  # type: ignore[attr-defined]
    if state is None:
        raise RuntimeError("Missing RT trace state.")
    return carry, state


def per_puzzle_hrm(
    model: torch.nn.Module, run: RunDirectory, x: torch.Tensor, max_l_updates: int, lag_points: int, progress: tqdm[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert run.l_cycles is not None
    blocks, native_l = max_l_updates // run.l_cycles, run.l_cycles
    boundaries = segment_boundaries(blocks)
    lags = np.asarray(segment_lags(boundaries, lag_points), dtype=np.int64)
    values = np.full((x.shape[0], len(STATE_NAMES), 4, len(lags)), np.nan, dtype=np.float32)
    origins = np.zeros((4, len(lags)), dtype=np.int32)
    original_h = model.H_cycles  # type: ignore[attr-defined]
    model.H_cycles, model.L_cycles = 1, native_l  # type: ignore[attr-defined]
    try:
        for lag_index, lag in enumerate(lags.tolist()):
            ref_carry: Carry = model.initial_carry  # type: ignore[attr-defined]
            lead_carry: Carry = model.initial_carry  # type: ignore[attr-defined]
            ref_state, lead_state = _initial_hrm(model, ref_carry, x), _initial_hrm(model, lead_carry, x)
            for _ in range(lag):
                lead_carry, lead_state = _advance_hrm(model, lead_carry, x)
            totals = {segment: {state: torch.zeros(x.shape[0], device=x.device) for state in STATE_NAMES} for segment in range(4)}
            counts = np.zeros(4, dtype=np.int32)
            for t in range(blocks - lag + 1):
                segment = segment_for_pair(t, lag, boundaries)
                if segment is not None:
                    for state, step_values in state_msd_per_puzzle(*ref_state, *lead_state).items():
                        totals[segment][state] += step_values
                    counts[segment] += 1
                if t < blocks - lag:
                    ref_carry, ref_state = _advance_hrm(model, ref_carry, x)
                    lead_carry, lead_state = _advance_hrm(model, lead_carry, x)
            for segment in range(4):
                if counts[segment]:
                    origins[segment, lag_index] = counts[segment]
                    for state_index, state in enumerate(STATE_NAMES):
                        values[:, state_index, segment, lag_index] = (totals[segment][state] / counts[segment]).cpu().numpy()
                    # With equal H/L state widths, the full concatenated state
                    # has per-coordinate MSD (MSD_H + MSD_L) / 2 exactly.  The
                    # independently accumulated CUDA reduction is equivalent in
                    # real arithmetic, but can differ by ~1e-5 after thousands
                    # of float32 additions.  Store the defining expression so
                    # downstream summaries retain the exact identity.
                    values[:, 3, segment, lag_index] = (
                        values[:, 0, segment, lag_index] + values[:, 1, segment, lag_index]
                    ) / 2
            progress.update(1)
    finally:
        model.H_cycles = original_h  # type: ignore[attr-defined]
    return values, lags, np.asarray(boundaries, dtype=np.int64), origins


def per_puzzle_rt(
    model: torch.nn.Module, x: torch.Tensor, max_updates: int, lag_points: int, progress: tqdm[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    boundaries = segment_boundaries(max_updates)
    lags = np.asarray(segment_lags(boundaries, lag_points), dtype=np.int64)
    values = np.full((x.shape[0], 1, 4, len(lags)), np.nan, dtype=np.float32)
    origins = np.zeros((4, len(lags)), dtype=np.int32)
    original_cycles = model.cycles  # type: ignore[attr-defined]
    model.cycles = 1  # type: ignore[attr-defined]
    try:
        for lag_index, lag in enumerate(lags.tolist()):
            ref_carry: Carry = model.initial_carry  # type: ignore[attr-defined]
            lead_carry: Carry = model.initial_carry  # type: ignore[attr-defined]
            initial = ref_carry["z"].reshape(1, 1, -1).expand(x.shape[0], x.shape[1], -1)
            ref_state, lead_state = initial, initial
            for _ in range(lag):
                lead_carry, lead_state = _advance_rt(model, lead_carry, x)
            totals = {segment: torch.zeros(x.shape[0], device=x.device) for segment in range(4)}
            counts = np.zeros(4, dtype=np.int32)
            for t in range(max_updates - lag + 1):
                segment = segment_for_pair(t, lag, boundaries)
                if segment is not None:
                    totals[segment] += rt_state_msd_per_puzzle(ref_state, lead_state)
                    counts[segment] += 1
                if t < max_updates - lag:
                    ref_carry, ref_state = _advance_rt(model, ref_carry, x)
                    lead_carry, lead_state = _advance_rt(model, lead_carry, x)
            for segment in range(4):
                if counts[segment]:
                    origins[segment, lag_index] = counts[segment]
                    values[:, 0, segment, lag_index] = (totals[segment] / counts[segment]).cpu().numpy()
            progress.update(1)
    finally:
        model.cycles = original_cycles  # type: ignore[attr-defined]
    return values, lags, np.asarray(boundaries, dtype=np.int64), origins


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(output, **arrays)
    temporary.replace(path)


def run_trajectory(
    run: RunDirectory, checkpoint: Path, best_epoch: int, fixed_x: torch.Tensor, stream_indices: np.ndarray,
    metadata: dict[str, Any], args: Any, manifest_sha256: str, progress: tqdm[Any],
) -> dict[str, object]:
    file_name = f"{run.condition}_seed_{run.seed}.npz"
    output_path = args.output_dir / "per_puzzle_msd" / file_name
    if output_path.is_file():
        cached = np.load(output_path, allow_pickle=False)
        matches = (
            "checkpoint" in cached.files
            and str(cached["checkpoint"].item()) == str(checkpoint)
            and np.array_equal(cached["sample_stream_indices"], stream_indices)
        )
        if matches:
            boundaries, lags = cached["segment_boundaries_blocks"], cached["lag_blocks"]
            return {
                "kind": run.kind, "condition": run.condition, "readout": run.readout, "train_l": "" if run.l_cycles is None else run.l_cycles,
                "seed": run.seed, "best_epoch": best_epoch, "checkpoint": str(checkpoint), "samples": len(stream_indices),
                "sample_seed": args.sample_seed, "sample_manifest_sha256": manifest_sha256,
                "requested_l_updates": args.max_l_updates, "actual_l_updates": int(boundaries[-1]) * (run.l_cycles or 1),
                "total_blocks": int(boundaries[-1]), "segment_boundaries_blocks": json.dumps(boundaries.tolist()),
                "lag_blocks": json.dumps(lags.tolist()), "state_shape": f"{len(stream_indices)}x82x512", "per_puzzle_file": str(output_path),
            }
    model = build_model(run, checkpoint, metadata, args.device)
    batches: list[np.ndarray] = []
    lags = boundaries = origins = None
    for start in range(0, fixed_x.shape[0], args.rollout_batch_size):
        x = fixed_x[start:start + args.rollout_batch_size].to(args.device, non_blocking=True)
        if run.kind == "hrm":
            values, current_lags, current_boundaries, current_origins = per_puzzle_hrm(model, run, x, args.max_l_updates, args.lag_points, progress)
        else:
            values, current_lags, current_boundaries, current_origins = per_puzzle_rt(model, x, args.max_l_updates, args.lag_points, progress)
        batches.append(values)
        if lags is None:
            lags, boundaries, origins = current_lags, current_boundaries, current_origins
        elif not (np.array_equal(lags, current_lags) and np.array_equal(boundaries, current_boundaries) and np.array_equal(origins, current_origins)):
            raise AssertionError("Chunked rollouts produced inconsistent metadata.")
    assert lags is not None and boundaries is not None and origins is not None
    all_values = np.concatenate(batches, axis=0)
    states = np.asarray(("rt",) if run.kind == "rt" else STATE_NAMES)
    if run.kind == "hrm":
        h, l, cat = all_values[:, 0], all_values[:, 1], all_values[:, 3]
        target = (h + l) / 2
        if not np.allclose(cat, target, rtol=0.0, atol=0.0, equal_nan=True):
            error = np.nanmax(np.abs(cat - target))
            raise RuntimeError(
                f"[H,L] identity failed for {run.condition}/seed_{run.seed}: "
                f"absolute={error:.3e}"
            )
    actual_updates = int(boundaries[-1]) * (run.l_cycles or 1)
    atomic_npz(output_path, msd=all_values, state_names=states, lag_blocks=lags,
               lag_l_updates=lags * (run.l_cycles or 1), segment_boundaries_blocks=boundaries,
               origins=origins, sample_stream_indices=stream_indices, checkpoint=np.asarray(str(checkpoint)))
    del model
    gc.collect()
    if args.device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "kind": run.kind, "condition": run.condition, "readout": run.readout, "train_l": "" if run.l_cycles is None else run.l_cycles,
        "seed": run.seed, "best_epoch": best_epoch, "checkpoint": str(checkpoint), "samples": len(stream_indices),
        "sample_seed": args.sample_seed, "sample_manifest_sha256": manifest_sha256,
        "requested_l_updates": args.max_l_updates, "actual_l_updates": actual_updates, "total_blocks": int(boundaries[-1]),
        "segment_boundaries_blocks": json.dumps(boundaries.tolist()), "lag_blocks": json.dumps(lags.tolist()),
        "state_shape": f"{len(stream_indices)}x82x512", "per_puzzle_file": str(output_path),
    }


def bootstrap(values: np.ndarray, indices: np.ndarray) -> tuple[float, float, float, float]:
    mean, median = float(np.mean(values)), float(np.median(values))
    means = np.mean(values[indices], axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return mean, median, float(low), float(high)


def summarize_npz_files(output_dir: Path, repeats: int, random_seed: int) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    metadata = read_csv(output_dir / "rollout_metadata.csv")
    rng = np.random.default_rng(random_seed)
    puzzle_rows: list[dict[str, object]] = []
    beta_rows: list[dict[str, object]] = []
    raw: dict[tuple[str, str, int], dict[str, Any]] = {}
    for meta in metadata:
        arrays = np.load(meta["per_puzzle_file"], allow_pickle=False)
        values = arrays["msd"].astype(np.float64)
        states = [str(value) for value in arrays["state_names"]]
        lags_b, lags_l, origins = arrays["lag_blocks"], arrays["lag_l_updates"], arrays["origins"]
        sample_count = values.shape[0]
        boot_indices = rng.integers(0, sample_count, size=(repeats, sample_count))
        raw[(meta["condition"], meta["readout"], int(meta["seed"]))] = {"meta": meta, "arrays": arrays}
        for state_index, state in enumerate(states):
            for segment in range(4):
                valid = np.isfinite(values[:, state_index, segment, :]).all(axis=0)
                if not np.any(valid):
                    continue
                segment_values = values[:, state_index, segment, valid]
                segment_lags_b, segment_lags_l = lags_b[valid], lags_l[valid]
                for point, (lag_b, lag_l) in enumerate(zip(segment_lags_b, segment_lags_l)):
                    mean, median, low, high = bootstrap(segment_values[:, point], boot_indices)
                    puzzle_rows.append({
                        "kind": meta["kind"], "condition": meta["condition"], "readout": meta["readout"], "train_l": meta["train_l"],
                        "seed": meta["seed"], "state": state, "segment": segment + 1, "lag_blocks": int(lag_b), "lag_l_updates": int(lag_l),
                        "mean": mean, "median": median, "ci95_low": low, "ci95_high": high,
                        "puzzles": sample_count, "origins": int(origins[segment, np.where(lags_b == lag_b)[0][0]]),
                    })
                beta = local_log_slope(segment_values, segment_lags_l)
                for point, (lag_b, lag_l) in enumerate(zip(segment_lags_b, segment_lags_l)):
                    mean, median, low, high = bootstrap(beta[:, point], boot_indices)
                    beta_rows.append({
                        "kind": meta["kind"], "condition": meta["condition"], "readout": meta["readout"], "train_l": meta["train_l"],
                        "seed": meta["seed"], "state": state, "segment": segment + 1, "lag_blocks": int(lag_b), "lag_l_updates": int(lag_l),
                        "beta_mean": mean, "beta_median": median, "ci95_low": low, "ci95_high": high, "puzzles": sample_count,
                    })
    seed_rows = cluster_seed_summary(raw, repeats, random_seed + 1)
    return puzzle_rows, seed_rows, beta_rows


def cluster_seed_summary(raw: dict[tuple[str, str, int], dict[str, Any]], repeats: int, random_seed: int) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for (condition, readout, _seed), item in raw.items():
        grouped.setdefault((condition, readout), []).append(item)
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, object]] = []
    for (_condition, _readout), items in grouped.items():
        items.sort(key=lambda item: int(item["meta"]["seed"]))
        first = items[0]
        first_values = first["arrays"]["msd"].astype(np.float64)
        states = [str(value) for value in first["arrays"]["state_names"]]
        for state_index, state in enumerate(states):
            for segment in range(4):
                for lag_index, lag_b in enumerate(first["arrays"]["lag_blocks"]):
                    seed_values = []
                    for item in items:
                        point = item["arrays"]["msd"][:, state_index, segment, lag_index].astype(np.float64)
                        if not np.isfinite(point).all():
                            continue
                        seed_values.append(point)
                    if not seed_values:
                        continue
                    point_means = np.asarray([point.mean() for point in seed_values])
                    # Cluster bootstrap: resample training seeds, then puzzles within each selected seed.
                    samples = np.empty(repeats, dtype=np.float64)
                    for replicate in range(repeats):
                        selected = rng.integers(0, len(seed_values), size=len(seed_values))
                        samples[replicate] = np.mean([
                            seed_values[index][rng.integers(0, len(seed_values[index]), size=len(seed_values[index]))].mean()
                            for index in selected
                        ])
                    meta = first["meta"]
                    rows.append({
                        "kind": meta["kind"], "condition": meta["condition"], "readout": meta["readout"], "train_l": meta["train_l"],
                        "state": state, "segment": segment + 1, "lag_blocks": int(lag_b),
                        "lag_l_updates": int(first["arrays"]["lag_l_updates"][lag_index]), "seed_mean": float(point_means.mean()),
                        "cluster_ci95_low": float(np.quantile(samples, .025)), "cluster_ci95_high": float(np.quantile(samples, .975)),
                        "seeds": len(seed_values), "puzzles_per_seed": len(seed_values[0]),
                    })
    return rows


def mobility_rows(metadata: list[dict[str, str]]) -> list[dict[str, object]]:
    rows = []
    for meta in metadata:
        if meta["kind"] != "hrm":
            continue
        arrays = np.load(meta["per_puzzle_file"], allow_pickle=False)
        values = arrays["msd"].astype(np.float64)
        for segment in range(4):
            h, l, summed = values[:, 0, segment, 0], values[:, 1, segment, 0], values[:, 2, segment, 0]
            rows.append({
                "condition": meta["condition"], "readout": meta["readout"], "seed": meta["seed"], "segment": segment + 1,
                "lag_blocks": int(arrays["lag_blocks"][0]), "lag_l_updates": int(arrays["lag_l_updates"][0]),
                "h_over_l": float(np.mean(h) / np.mean(l)), "interaction": float(np.mean((summed - h - l) / 2)),
                "interaction_fraction_h_plus_l": float(np.mean((summed - h - l) / 2) / np.mean(summed)),
            })
    return rows


def condition_curve(metadata: list[dict[str, str]], condition: str, state: str, segment: int, repeats: int, rng: np.random.Generator) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = sorted([meta for meta in metadata if meta["condition"] == condition], key=lambda meta: int(meta["seed"]))
    arrays = [np.load(meta["per_puzzle_file"], allow_pickle=False) for meta in selected]
    actual_state = "rt" if state == "joint_or_rt" and condition == "RT" else ("hl_concat" if state == "joint_or_rt" else state)
    state_index = [str(value) for value in arrays[0]["state_names"]].index(actual_state)
    lags = arrays[0]["lag_l_updates"].astype(np.float64)
    seed_curves = []
    per_seed = []
    for data in arrays:
        values = data["msd"][:, state_index, segment, :].astype(np.float64)
        valid = np.isfinite(values).all(axis=0)
        seed_curves.append((lags[valid], values[:, valid].mean(axis=0)))
        per_seed.append(values)
    valid = np.logical_and.reduce([np.isfinite(values).all(axis=0) for values in per_seed])
    lags = lags[valid]
    per_seed = [values[:, valid] for values in per_seed]
    mean = np.mean([values.mean(axis=0) for values in per_seed], axis=0)
    ci_low, ci_high = [], []
    for point in range(len(lags)):
        samples = []
        for _ in range(repeats):
            chosen = rng.integers(0, len(per_seed), size=len(per_seed))
            samples.append(np.mean([per_seed[index][rng.integers(0, per_seed[index].shape[0], per_seed[index].shape[0]), point].mean() for index in chosen]))
        ci_low.append(np.quantile(samples, .025)); ci_high.append(np.quantile(samples, .975))
    return seed_curves, lags, mean, np.asarray(ci_low), np.asarray(ci_high)


def plot_comparison(output_dir: Path, metadata: list[dict[str, str]], conditions: tuple[str, ...], states: tuple[str, ...], name: str, title: str, repeats: int, seed: int) -> None:
    figure, axes = plt.subplots(len(states), 4, figsize=(15, 3.4 * len(states)), sharex=False, sharey="row")
    if len(states) == 1:
        axes = np.asarray([axes])
    colors = plt.get_cmap("tab10")
    rng = np.random.default_rng(seed)
    for row, state in enumerate(states):
        for segment in range(4):
            axis = axes[row, segment]
            for index, condition in enumerate(conditions):
                seed_curves, lags, mean, low, high = condition_curve(metadata, condition, state, segment, repeats, rng)
                color = colors(index)
                for seed_lags, seed_values in seed_curves:
                    axis.plot(seed_lags, seed_values, color=color, alpha=.18, linewidth=.8)
                axis.plot(lags, mean, color=color, linewidth=2, label=condition)
                axis.fill_between(lags, low, high, color=color, alpha=.16)
            axis.set_xscale("log", base=2); axis.set_yscale("log", base=2)
            axis.set_title(f"{state}, segment {segment + 1}")
            axis.grid(True, which="both", alpha=.2)
            if row == len(states) - 1:
                axis.set_xlabel("Lag (underlying updates, log₂)")
            if segment == 0:
                axis.set_ylabel("Per-coordinate MSD")
            if row == 0 and segment == 3:
                axis.legend(fontsize=8)
    figure.suptitle(title)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"{name}.{suffix}", dpi=200)
    plt.close(figure)


def create_figures(output_dir: Path, metadata: list[dict[str, str]], repeats: int, seed: int) -> None:
    plot_comparison(output_dir, metadata, ("H2L1_h", "H2L6_h"), ("h", "l"), "compare_l_refinement", "L refinement depth: H2L1-H vs H2L6-H", repeats, seed)
    plot_comparison(output_dir, metadata, ("H2L6_h", "H2L6_l", "H2L6_hl"), ("h", "l"), "compare_readout", "Readout supervision at H2L6", repeats, seed + 1)
    plot_comparison(output_dir, metadata, ("H2L1_h", "H2L6_h", "RT"), ("joint_or_rt",), "compare_rt", "RT control vs HRM joint state", repeats, seed + 2)


def write_readme(output_dir: Path) -> None:
    (output_dir / "README.md").write_text("""# Core-five statistical long-rollout analysis

This analysis covers H2L1-H, H2L6-H, H2L6-L, H2L6-HL, and RT for seeds 1, 2, and 3.
Best checkpoints are selected independently per condition and seed by native-schedule
test_hard exact match. The training-compatible test loader uses drop_last=True, so
selection evaluates 19,968 effective examples.

The 256 rollout puzzles are a fixed random subset of this same effective test stream;
their indices and hash are in sample_manifest.csv. Per-puzzle tensors are stored in
per_puzzle_msd/*.npz. Each tensor has [puzzle, state, segment, lag] axes and is the
time-averaged full-state per-coordinate MSD. Bootstrap intervals resample puzzles;
the cross-seed summary uses a nested cluster bootstrap over training seeds and puzzles.

The four segments remain log-uniform in each model's H-boundary clock. HRM states are
sampled after each H update, so L is observed at L-block boundaries, not after every
individual L update. The RT comparison uses HRM [H,L] and the RT state, both normalized
per coordinate. The figures are descriptive deterministic transport analyses, not
evidence that inference is a stochastic loss-landscape diffusion process.
""")


def merge_core(args: Any) -> None:
    source_dirs = args.merge_from
    manifests = [Path(directory) / "sample_manifest.sha256" for directory in source_dirs]
    hashes = [path.read_text().strip() for path in manifests]
    if len(set(hashes)) != 1:
        raise ValueError("Worker sample manifests differ; refusing to merge incomparable puzzle sets.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_best: list[dict[str, str]] = []
    all_metadata: list[dict[str, str]] = []
    for source in source_dirs:
        all_best.extend(read_csv(Path(source) / "best_checkpoints.csv"))
        all_metadata.extend(read_csv(Path(source) / "rollout_metadata.csv"))
        for source_npz in (Path(source) / "per_puzzle_msd").glob("*.npz"):
            target = args.output_dir / "per_puzzle_msd" / source_npz.name
            if target.exists():
                raise ValueError(f"Duplicate per-puzzle result: {target.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_npz, target)
    expected = {(condition, seed) for condition in CORE_CONDITIONS for seed in args.core_seeds}
    actual = {(row["condition"], int(row["seed"])) for row in all_metadata}
    if actual != expected:
        raise ValueError(f"Merged workers incomplete: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    for row in all_metadata:
        row["per_puzzle_file"] = str(args.output_dir / "per_puzzle_msd" / Path(row["per_puzzle_file"]).name)
    atomic_csv(args.output_dir / "best_checkpoints.csv", all_best, SELECTION_FIELDS)
    atomic_csv(args.output_dir / "accuracy_core.csv", all_best, SELECTION_FIELDS)
    atomic_csv(args.output_dir / "rollout_metadata.csv", all_metadata, ROLLOUT_FIELDS)
    shutil.copy2(Path(source_dirs[0]) / "sample_manifest.csv", args.output_dir / "sample_manifest.csv")
    shutil.copy2(manifests[0], args.output_dir / "sample_manifest.sha256")
    finalize_core(args)


def finalize_core(args: Any) -> None:
    metadata = read_csv(args.output_dir / "rollout_metadata.csv")
    puzzle_rows, seed_rows, beta_rows = summarize_npz_files(args.output_dir, args.bootstrap_replicates, args.sample_seed)
    atomic_csv(args.output_dir / "msd_puzzle_bootstrap.csv", puzzle_rows, PUZZLE_SUMMARY_FIELDS)
    atomic_csv(args.output_dir / "msd_seed_cluster_bootstrap.csv", seed_rows, SEED_SUMMARY_FIELDS)
    atomic_csv(args.output_dir / "beta_puzzle_bootstrap.csv", beta_rows, BETA_FIELDS)
    atomic_csv(args.output_dir / "mobility_summary.csv", mobility_rows(metadata), MOBILITY_FIELDS)
    create_figures(args.output_dir, metadata, args.bootstrap_replicates, args.sample_seed)
    (args.output_dir / "analysis_metadata.json").write_text(json.dumps({
        "profile": "core-five", "conditions": CORE_CONDITIONS, "seeds": args.core_seeds,
        "samples": args.samples, "sample_seed": args.sample_seed, "bootstrap_replicates": args.bootstrap_replicates,
        "max_l_updates": args.max_l_updates, "lag_points": args.lag_points,
    }, indent=2) + "\n")
    write_readme(args.output_dir)


def main_core(args: Any) -> None:
    if args.merge_from:
        merge_core(args)
        return
    runs = core_runs(args.checkpoints_root, args.core_seeds)
    if args.shard_index is not None:
        runs = runs[args.shard_index::args.num_shards]
    if not runs:
        raise ValueError("This core-five shard received no runs.")
    first_config = runs[0].config
    create_dataloader = load_module(f"dataset.{first_config.data.name}@create_dataloader")
    test_loader, metadata = create_dataloader("test_hard", first_config.local_batch_size, rank=0, world_size=1, **data_kwargs(first_config))
    best = select_best_resumable(runs, test_loader, metadata, args.device, args.output_dir)
    fixed_x, indices = fixed_random_samples(test_loader, args.samples, args.sample_seed)
    manifest = write_manifest(args.output_dir, indices, args.sample_seed)
    by_key = {(run.condition, run.seed): run for run in runs}
    chunks = math.ceil(args.samples / args.rollout_batch_size)
    total = sum(len(segment_lags(segment_boundaries(args.max_l_updates // (run.l_cycles or 1)), args.lag_points)) * chunks for run in runs)
    progress = tqdm(total=total, desc="Core per-puzzle paired rollouts", unit="lag-batch")
    rollout_rows = []
    for row in best:
        run = by_key[(str(row["condition"]), int(row["seed"]))]
        progress.set_postfix_str(f"{run.condition}/seed_{run.seed}")
        rollout_rows.append(run_trajectory(run, Path(str(row["checkpoint"])), int(row["epoch"]), fixed_x, indices, metadata, args, manifest, progress))
        atomic_csv(args.output_dir / "rollout_metadata.csv", rollout_rows, ROLLOUT_FIELDS)
    progress.close()
    atomic_csv(args.output_dir / "accuracy_core.csv", best, SELECTION_FIELDS)
    if args.num_shards == 1:
        finalize_core(args)


__all__ = ["CORE_CONDITIONS", "core_runs", "main_core", "merge_core", "fixed_random_samples"]

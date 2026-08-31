"""Evaluate the complete HRM/RT/TRM Sudoku generalization matrix.

The primary result intentionally uses the final checkpoint for every run.
Per-epoch held-out curves are useful diagnostics, but selecting their peak
would tune on the target test band and invalidate a strict generalization
claim.

Example:
    python cross_evaluate.py \\
      --checkpoint hrm:easy:1=checkpoints/easy-hrm/seed_1/epoch_19.pt \\
      --checkpoint rt:easy:1=checkpoints/easy-rt/seed_1/epoch_19.pt \\
      --checkpoint trm:easy:1=checkpoints/easy-trm/seed_1/epoch_19.pt \\
      ...
"""

import argparse
import csv
import importlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

MODELS = ("hrm", "rt", "trm")
TRAIN_BANDS = ("easy", "medium", "hard")
DIFFICULTY_BOUNDS: dict[str, tuple[int | None, int | None]] = {
    "easy": (0, 15),
    "medium": (16, 30),
    "hard": (51, None),
}
EVAL_SAMPLE_SIZE = 10_000
EVAL_SEED = 42


@dataclass(frozen=True)
class ExperimentCheckpoint:
    model: str
    train_band: str
    seed: int
    path: Path


def _load_checkpoint_config(checkpoint: Path) -> dict[str, Any]:
    config_path = checkpoint.parent / "model_config.json"
    if not config_path.is_file():
        raise ValueError(f"missing model_config.json next to checkpoint: {checkpoint}")
    with config_path.open() as f:
        raw_config = yaml.safe_load(f)
    if not isinstance(raw_config, dict) or "arch" not in raw_config or "data" not in raw_config:
        raise ValueError(f"invalid model configuration next to checkpoint: {checkpoint}")
    return raw_config


def _load_module(identifier: str):
    module_path, class_name = identifier.split("@")
    return getattr(importlib.import_module(module_path), class_name)


def parse_checkpoint(value: str) -> ExperimentCheckpoint:
    """Parse MODEL:TRAIN_BAND:SEED=PATH and validate static run metadata."""
    identifier, separator, checkpoint = value.partition("=")
    parts = identifier.split(":")
    if separator != "=" or len(parts) != 3 or not checkpoint:
        raise argparse.ArgumentTypeError("checkpoint must be MODEL:TRAIN_BAND:SEED=PATH")
    model, train_band, seed_text = parts
    if model not in MODELS:
        raise argparse.ArgumentTypeError(f"model must be one of: {', '.join(MODELS)}")
    if train_band not in TRAIN_BANDS:
        raise argparse.ArgumentTypeError(f"train band must be one of: {', '.join(TRAIN_BANDS)}")
    try:
        seed = int(seed_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("seed must be an integer") from error
    if seed < 0:
        raise argparse.ArgumentTypeError("seed must be non-negative")

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise argparse.ArgumentTypeError(f"checkpoint does not exist: {checkpoint_path}")
    try:
        config = _load_checkpoint_config(checkpoint_path)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error

    architecture_class = str(config["arch"]["name"]).rsplit("@", 1)[-1]
    expected_classes = {"hrm": "HRM", "rt": "RecurrentTransformer", "trm": "TRM"}
    if architecture_class != expected_classes[model]:
        raise argparse.ArgumentTypeError(
            f"{checkpoint_path} contains {architecture_class}, not the declared {model} architecture"
        )
    final_filename = f"epoch_{int(config['epochs']) - 1}.pt"
    if checkpoint_path.name != final_filename:
        raise argparse.ArgumentTypeError(
            f"primary results require the final checkpoint {final_filename}, got {checkpoint_path.name}"
        )
    return ExperimentCheckpoint(model, train_band, seed, checkpoint_path)


def validate_experiment_matrix(
    checkpoints: list[ExperimentCheckpoint], models: tuple[str, ...] = MODELS
) -> list[int]:
    """Require a complete, balanced selected-model x train-band x seed matrix."""
    by_key: dict[tuple[str, str, int], ExperimentCheckpoint] = {}
    for checkpoint in checkpoints:
        key = (checkpoint.model, checkpoint.train_band, checkpoint.seed)
        if key in by_key:
            raise ValueError(f"duplicate checkpoint supplied for {checkpoint.model}:{checkpoint.train_band}:{checkpoint.seed}")
        by_key[key] = checkpoint

    expected_pairs = {(model, band) for model in models for band in TRAIN_BANDS}
    actual_pairs = {(item.model, item.train_band) for item in checkpoints}
    missing_pairs = expected_pairs - actual_pairs
    unexpected_pairs = actual_pairs - expected_pairs
    if missing_pairs or unexpected_pairs:
        missing_text = ", ".join(f"{m}:{b}" for m, b in sorted(missing_pairs))
        raise ValueError(f"complete comparison requires every selected model/train band pair; missing: {missing_text}")

    seed_sets = {
        pair: {item.seed for item in checkpoints if (item.model, item.train_band) == pair}
        for pair in expected_pairs
    }
    reference_seeds = next(iter(seed_sets.values()))
    if not reference_seeds or any(seeds != reference_seeds for seeds in seed_sets.values()):
        raise ValueError("every model/train-band pair must provide the identical non-empty seed set")
    return sorted(reference_seeds)


def _budget_metadata(checkpoint: ExperimentCheckpoint) -> dict[str, Any]:
    config = _load_checkpoint_config(checkpoint.path)
    # Sudoku's model metadata is fixed by dataset/sudoku.py. Instantiate on
    # CPU only to count trainable parameters; evaluation performs the GPU load.
    arch_options = dict(config["arch"])
    arch_name = str(arch_options.pop("name"))
    model_cls = _load_module(f"arch.{arch_name}")
    model = model_cls(arch_options | {
        "vocab_size": 10,
        "seq_len": 82,
        "is_causal": False,
    })
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    if "H_cycles" in arch_options and "L_cycles" in arch_options:
        core_calls_per_forward = int(arch_options["H_cycles"]) * (int(arch_options["L_cycles"]) + 1)
    elif "cycles" in arch_options:
        core_calls_per_forward = int(arch_options["cycles"])
    else:
        core_calls_per_forward = None

    run_metadata = config.get("run_metadata", {})
    return {
        "parameter_count": parameter_count,
        "core_calls_per_model_forward": core_calls_per_forward,
        "core_calls_per_prediction": (
            None if core_calls_per_forward is None else core_calls_per_forward * int(config["cycles_per_data"])
        ),
        "cycles_per_data": int(config["cycles_per_data"]),
        "epochs": int(config["epochs"]),
        "total_training_steps_per_rank": run_metadata.get("total_training_steps_per_rank"),
        "world_size": run_metadata.get("world_size"),
    }


def _selected_eval_dataset(checkpoint: ExperimentCheckpoint, override: str | None) -> str:
    config = _load_checkpoint_config(checkpoint.path)
    data_options = dict(config["data"])
    return override or str(data_options.get("eval_dataset_name", data_options.get("dataset_name")))


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["train_band"], row["test_band"])].append(row)

    aggregates = []
    for (model, train_band, test_band), group in sorted(grouped.items()):
        accuracies = np.array([row["exact_match_accuracy"] for row in group], dtype=float)
        aggregates.append({
            "model": model,
            "train_band": train_band,
            "test_band": test_band,
            "num_seeds": len(group),
            "exact_match_accuracy_mean": float(accuracies.mean()),
            "exact_match_accuracy_std": float(accuracies.std(ddof=1)) if len(group) > 1 else 0.0,
            "total_samples_per_seed": group[0]["total_samples"],
            "parameter_count": group[0]["parameter_count"],
            "core_calls_per_prediction": group[0]["core_calls_per_prediction"],
            "total_training_steps_per_rank": group[0]["total_training_steps_per_rank"],
            "world_size": group[0]["world_size"],
        })
    return aggregates


def write_results(output_dir: Path, rows: list[dict[str, Any]], aggregates: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_fields = list(rows[0])
    aggregate_fields = list(aggregates[0])
    with (output_dir / "cross_evaluation_runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "cross_evaluation_aggregate.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregates)
    with (output_dir / "cross_evaluation_runs.json").open("w") as f:
        json.dump(rows, f, indent=2)
    with (output_dir / "cross_evaluation_aggregate.json").open("w") as f:
        json.dump(aggregates, f, indent=2)

    matrices: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for aggregate in aggregates:
        matrices.setdefault(aggregate["model"], {}).setdefault(aggregate["train_band"], {})[
            aggregate["test_band"]
        ] = {
            "mean": aggregate["exact_match_accuracy_mean"],
            "std": aggregate["exact_match_accuracy_std"],
        }
    with (output_dir / "cross_evaluation_matrices.json").open("w") as f:
        json.dump(matrices, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the final HRM/RT/TRM cross-difficulty matrix.")
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_checkpoint,
        required=True,
        metavar="MODEL:TRAIN_BAND:SEED=PATH",
        help="Repeat for every HRM/RT/TRM x easy/medium/hard x seed final checkpoint.",
    )
    parser.add_argument("--eval-dataset-name", help="Optional common held-out Sudoku dataset override.")
    parser.add_argument(
        "--models", nargs="+", choices=MODELS, default=list(MODELS),
        help="Models included in the complete matrix (default: hrm rt trm).",
    )
    parser.add_argument("--output-dir", default="results/cross_evaluation", help="Directory for metrics and correctness files")
    args = parser.parse_args()

    checkpoints: list[ExperimentCheckpoint] = args.checkpoint
    selected_models = tuple(args.models)
    if len(set(selected_models)) != len(selected_models):
        parser.error("--models may not contain duplicates")
    if any(checkpoint.model not in selected_models for checkpoint in checkpoints):
        parser.error("every supplied checkpoint model must be listed in --models")
    seeds = validate_experiment_matrix(checkpoints, models=selected_models)
    selected_datasets = {_selected_eval_dataset(checkpoint, args.eval_dataset_name) for checkpoint in checkpoints}
    if len(selected_datasets) != 1:
        raise ValueError(
            "all checkpoints must use the same held-out evaluation dataset; "
            "pass --eval-dataset-name to set one explicitly"
        )

    output_dir = Path(args.output_dir)
    # Import the GPU evaluation stack only for an actual evaluation run. The
    # parser/aggregator can consequently be used in lightweight environments.
    from eval import evaluate_checkpoint

    rows: list[dict[str, Any]] = []
    for checkpoint in sorted(checkpoints, key=lambda item: (item.model, item.train_band, item.seed)):
        budget = _budget_metadata(checkpoint)
        for test_band, (rating_min, rating_max) in DIFFICULTY_BOUNDS.items():
            print(f"\n=== model={checkpoint.model}, train={checkpoint.train_band}, seed={checkpoint.seed}, test={test_band} ===")
            metrics = evaluate_checkpoint(
                str(checkpoint.path),
                split="test",
                eval_rating_min=rating_min,
                eval_rating_max=rating_max,
                eval_dataset_name=args.eval_dataset_name,
                eval_num_base_puzzles=EVAL_SAMPLE_SIZE,
                eval_seed=EVAL_SEED,
                output=str(output_dir / (
                    f"model_{checkpoint.model}_train_{checkpoint.train_band}_seed_{checkpoint.seed}_test_{test_band}.npz"
                )),
            )
            rows.append({
                "model": checkpoint.model,
                "train_band": checkpoint.train_band,
                "seed": checkpoint.seed,
                "test_band": test_band,
                "rating_min": rating_min,
                "rating_max": rating_max,
                "checkpoint": str(checkpoint.path),
                "eval_dataset_name": next(iter(selected_datasets)),
                "eval_num_base_puzzles": EVAL_SAMPLE_SIZE,
                "eval_seed": EVAL_SEED,
                **metrics,
                **budget,
            })

    aggregates = aggregate_rows(rows)
    write_results(output_dir, rows, aggregates)
    print(f"\nEvaluated {len(rows)} cells across seeds {seeds}; results saved to {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Evaluate every H=2 L/readout sweep checkpoint across a common L-evaluation grid.

For each trained checkpoint, the script evaluates test exact-match accuracy at every
configured L value and emits one accuracy-vs-L plot per readout. It additionally
evaluates the checkpoint at its native L value on the unaugmented, non-repeated train
split, producing a directly comparable train/test table.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import sys
from typing import Any

# `python scripts/evaluate_h2_l_readout_sweep.py` adds only scripts/ to sys.path.
# Add the repository root so the sibling arch/ and train.py modules resolve reliably.
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

from arch.layers import Carry
from train import TrainConfig, load_module, run_inference


H_CYCLES = 2
TRAIN_L_VALUES = (1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32)
DEFAULT_EVAL_L_VALUES = tuple(range(1, 33))
READOUTS = ("h", "l", "hl")


def checkpoint_path(checkpoints_root: Path, train_l: int, readout: str, seed: int, epoch: int) -> Path:
    return checkpoints_root / f"H2L{train_l}_{readout}" / f"seed_{seed}" / f"epoch_{epoch}.pt"


def data_kwargs(config: TrainConfig, *, augment: bool, repeat: int) -> dict[str, Any]:
    kwargs = dict(config.data.__pydantic_extra__ or {})
    kwargs["augment"] = augment
    kwargs["repeat"] = repeat
    return kwargs


@torch.inference_mode()
def exact_match_accuracy(model: torch.nn.Module, loader: Any, cycles_per_data: int, eval_l: int) -> float:
    model.H_cycles = H_CYCLES  # type: ignore[attr-defined]
    model.L_cycles = eval_l  # type: ignore[attr-defined]

    total_correct = 0
    total_samples = 0
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        carry: Carry = model.initial_carry  # type: ignore[attr-defined]
        y_hat = None
        for _ in range(cycles_per_data):
            carry, y_hat = run_inference(model, carry, x)

        total_correct += torch.all(y_hat == y, dim=-1).sum().item()
        total_samples += y.shape[0]

    if total_samples == 0:
        raise RuntimeError("Evaluation loader produced no samples.")
    return total_correct / total_samples


def load_config(checkpoint: Path) -> TrainConfig:
    with (checkpoint.parent / "model_config.json").open() as config_file:
        return TrainConfig(**yaml.safe_load(config_file))


def validate_checkpoint_config(config: TrainConfig, train_l: int, readout: str) -> None:
    arch = config.arch.__pydantic_extra__ or {}
    if arch.get("H_cycles") != H_CYCLES:
        raise ValueError(f"Expected H_cycles={H_CYCLES}, got {arch.get('H_cycles')}.")
    if arch.get("L_cycles") != train_l:
        raise ValueError(f"Expected train L={train_l}, got {arch.get('L_cycles')}.")
    if arch.get("readout") != readout:
        raise ValueError(f"Expected readout={readout!r}, got {arch.get('readout')!r}.")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_native_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "| readout | train L | train exact match | test exact match |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['readout']} | {row['train_l']} | "
            f"{float(row['train_exact_match']):.4f} | {float(row['test_exact_match']):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def plot_curves(output_dir: Path, curve_rows: list[dict[str, object]]) -> None:
    for readout in READOUTS:
        figure, axis = plt.subplots(figsize=(8, 5))
        for train_l in TRAIN_L_VALUES:
            points = [
                row for row in curve_rows
                if row["readout"] == readout and row["train_l"] == train_l
            ]
            points.sort(key=lambda row: int(row["eval_l"]))
            axis.plot(
                [int(row["eval_l"]) for row in points],
                [float(row["test_exact_match"]) for row in points],
                marker="o",
                linewidth=1.5,
                markersize=3,
                label=f"train L={train_l}",
            )

        axis.set_xscale("log", base=2)
        axis.set_xticks([1, 2, 4, 8, 16, 32])
        axis.set_xticklabels(["1", "2", "4", "8", "16", "32"])
        axis.set_xlabel("Evaluation L cycles (H=2)")
        axis.set_ylabel("Test exact-match accuracy")
        axis.set_title(f"H=2 L generalization — {readout.upper()} readout")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(ncol=2, fontsize=8, title="Training schedule")
        figure.tight_layout()
        figure.savefig(output_dir / f"accuracy_vs_eval_l_{readout}.png", dpi=200)
        plt.close(figure)


def parse_l_values(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(sorted({int(item) for item in value.split(",") if item}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("L values must be comma-separated positive integers.") from error
    if not parsed or parsed[0] < 1:
        raise argparse.ArgumentTypeError("Provide at least one positive L value.")
    return parsed


def parse_readouts(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = sorted(set(parsed) - set(READOUTS))
    if not parsed or invalid:
        raise argparse.ArgumentTypeError(f"Readouts must be a comma-separated subset of {','.join(READOUTS)}.")
    return tuple(readout for readout in READOUTS if readout in parsed)


def selected_conditions(
    readouts: tuple[str, ...], train_l_values: tuple[int, ...], shard_index: int | None, num_shards: int,
) -> list[tuple[str, int]]:
    conditions = [(readout, train_l) for readout in readouts for train_l in train_l_values]
    if shard_index is None:
        if num_shards != 1:
            raise ValueError("--num-shards requires --shard-index.")
        return conditions
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"--shard-index must be in [0, {num_shards - 1}].")
    return conditions[shard_index::num_shards]


def parse_curve_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as source:
        return [
            {
                "readout": row["readout"],
                "train_l": int(row["train_l"]),
                "eval_l": int(row["eval_l"]),
                "test_exact_match": float(row["test_exact_match"]),
            }
            for row in csv.DictReader(source)
        ]


def parse_native_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as source:
        return [
            {
                "readout": row["readout"],
                "train_l": int(row["train_l"]),
                "train_exact_match": float(row["train_exact_match"]),
                "test_exact_match": float(row["test_exact_match"]),
            }
            for row in csv.DictReader(source)
        ]


def validate_complete_results(
    curve_rows: list[dict[str, object]], native_rows: list[dict[str, object]], eval_l_values: tuple[int, ...],
) -> None:
    expected_curves = {(readout, train_l, eval_l) for readout in READOUTS for train_l in TRAIN_L_VALUES for eval_l in eval_l_values}
    actual_curves = [(str(row["readout"]), int(row["train_l"]), int(row["eval_l"])) for row in curve_rows]
    duplicate_curves = len(actual_curves) != len(set(actual_curves))
    missing_curves = expected_curves - set(actual_curves)
    extra_curves = set(actual_curves) - expected_curves
    expected_native = {(readout, train_l) for readout in READOUTS for train_l in TRAIN_L_VALUES}
    actual_native = [(str(row["readout"]), int(row["train_l"])) for row in native_rows]
    duplicate_native = len(actual_native) != len(set(actual_native))
    missing_native = expected_native - set(actual_native)
    extra_native = set(actual_native) - expected_native
    if duplicate_curves or missing_curves or extra_curves or duplicate_native or missing_native or extra_native:
        raise ValueError(
            "Shard results are incomplete or overlap: "
            f"missing curves={len(missing_curves)}, extra curves={len(extra_curves)}, duplicate curves={duplicate_curves}; "
            f"missing native rows={len(missing_native)}, extra native rows={len(extra_native)}, duplicate native rows={duplicate_native}."
        )


def write_final_results(
    output_dir: Path, curve_rows: list[dict[str, object]], native_rows: list[dict[str, object]], eval_l_values: tuple[int, ...],
    seed: int, epoch: int,
) -> None:
    curve_rows.sort(key=lambda row: (str(row["readout"]), int(row["train_l"]), int(row["eval_l"])))
    native_rows.sort(key=lambda row: (str(row["readout"]), int(row["train_l"])))
    write_csv(
        output_dir / "test_accuracy_curves.csv",
        curve_rows,
        ["readout", "train_l", "eval_l", "test_exact_match"],
    )
    write_csv(
        output_dir / "native_train_test_accuracy.csv",
        native_rows,
        ["readout", "train_l", "train_exact_match", "test_exact_match"],
    )
    write_native_markdown(output_dir / "native_train_test_accuracy.md", native_rows)
    plot_curves(output_dir, curve_rows)
    (output_dir / "evaluation_metadata.json").write_text(json.dumps({
        "h_cycles": H_CYCLES,
        "train_l_values": TRAIN_L_VALUES,
        "eval_l_values": eval_l_values,
        "readouts": READOUTS,
        "seed": seed,
        "epoch": epoch,
        "train_accuracy_definition": "exact match on train split with augment=False and repeat=1",
        "test_split": "test_hard",
    }, indent=2) + "\n")


def merge_shards(input_dirs: list[Path], output_dir: Path, eval_l_values: tuple[int, ...], seed: int, epoch: int) -> None:
    curve_rows: list[dict[str, object]] = []
    native_rows: list[dict[str, object]] = []
    for input_dir in input_dirs:
        curve_path = input_dir / "test_accuracy_curves.csv"
        native_path = input_dir / "native_train_test_accuracy.csv"
        if not curve_path.is_file() or not native_path.is_file():
            raise FileNotFoundError(f"Missing worker CSVs in {input_dir}.")
        curve_rows.extend(parse_curve_rows(curve_path))
        native_rows.extend(parse_native_rows(native_path))
    validate_complete_results(curve_rows, native_rows, eval_l_values)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_final_results(output_dir, curve_rows, native_rows, eval_l_values, seed, epoch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-root", type=Path, default=Path("checkpoints/h2_l_readout_sweep"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/h2_l_readout_sweep"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=19)
    parser.add_argument(
        "--eval-l-values",
        type=parse_l_values,
        default=DEFAULT_EVAL_L_VALUES,
        help="Comma-separated inference L values (default: every integer from 1 through 32).",
    )
    parser.add_argument("--readouts", type=parse_readouts, default=READOUTS, help="Comma-separated readout subset to evaluate.")
    parser.add_argument("--train-l-values", type=parse_l_values, default=TRAIN_L_VALUES, help="Comma-separated trained-L subset to evaluate.")
    parser.add_argument("--shard-index", type=int, help="Zero-based worker index used with --num-shards.")
    parser.add_argument("--num-shards", type=int, default=1, help="Split the selected conditions across this many workers.")
    parser.add_argument(
        "--merge-from", type=Path, nargs="+", metavar="WORKER_DIR",
        help="Merge complete worker CSVs from these directories instead of evaluating checkpoints.",
    )
    args = parser.parse_args()

    eval_l_values = tuple(sorted(set(args.eval_l_values) | set(TRAIN_L_VALUES)))
    if args.num_shards < 1:
        parser.error("--num-shards must be positive.")
    invalid_train_l_values = sorted(set(args.train_l_values) - set(TRAIN_L_VALUES))
    if invalid_train_l_values:
        parser.error(f"--train-l-values contains values outside the sweep: {invalid_train_l_values}")
    if args.merge_from:
        merge_shards(args.merge_from, args.output_dir, eval_l_values, args.seed, args.epoch)
        return

    conditions = selected_conditions(args.readouts, args.train_l_values, args.shard_index, args.num_shards)
    if not conditions:
        parser.error("This worker received no conditions.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    expected_checkpoints = [
        checkpoint_path(args.checkpoints_root, train_l, readout, args.seed, args.epoch)
        for readout, train_l in conditions
    ]
    missing = [path for path in expected_checkpoints if not path.is_file()]
    if missing:
        formatted_paths = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing {len(missing)} required checkpoints:\n{formatted_paths}")

    first_config = load_config(expected_checkpoints[0])
    create_dataloader = load_module(f"dataset.{first_config.data.name}@create_dataloader")
    test_loader, metadata = create_dataloader(
        "test_hard", first_config.local_batch_size, rank=0, world_size=1,
        **data_kwargs(first_config, augment=False, repeat=1),
    )
    train_loader, _ = create_dataloader(
        "train", first_config.local_batch_size, rank=0, world_size=1,
        **data_kwargs(first_config, augment=False, repeat=1),
    )

    curve_rows: list[dict[str, object]] = []
    native_rows: list[dict[str, object]] = []
    total_evaluations = len(conditions) * (len(eval_l_values) + 1)
    progress = tqdm(total=total_evaluations, desc="Dataset evaluations", unit="pass")
    for readout, train_l in conditions:
        checkpoint = checkpoint_path(args.checkpoints_root, train_l, readout, args.seed, args.epoch)
        config = load_config(checkpoint)
        validate_checkpoint_config(config, train_l, readout)

        progress.set_postfix_str(f"{readout}, train L={train_l}")
        model_cls = load_module(f"arch.{config.arch.name}")
        with torch.device("cuda"):
            model = model_cls((config.arch.__pydantic_extra__ or {}) | metadata)
            model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True), assign=True)
        model.eval()

        native_test_accuracy: float | None = None
        for eval_l in eval_l_values:
            test_accuracy = exact_match_accuracy(model, test_loader, config.cycles_per_data, eval_l)
            curve_rows.append({
                "readout": readout,
                "train_l": train_l,
                "eval_l": eval_l,
                "test_exact_match": test_accuracy,
            })
            progress.update(1)
            if eval_l == train_l:
                native_test_accuracy = test_accuracy

        train_accuracy = exact_match_accuracy(model, train_loader, config.cycles_per_data, train_l)
        progress.update(1)
        assert native_test_accuracy is not None
        native_rows.append({
            "readout": readout,
            "train_l": train_l,
            "train_exact_match": train_accuracy,
            "test_exact_match": native_test_accuracy,
        })
        del model
        gc.collect()
        torch.cuda.empty_cache()
    progress.close()

    write_csv(
        args.output_dir / "test_accuracy_curves.csv",
        curve_rows,
        ["readout", "train_l", "eval_l", "test_exact_match"],
    )
    write_csv(
        args.output_dir / "native_train_test_accuracy.csv",
        native_rows,
        ["readout", "train_l", "train_exact_match", "test_exact_match"],
    )
    (args.output_dir / "worker_metadata.json").write_text(json.dumps({
        "readouts": args.readouts,
        "train_l_values": args.train_l_values,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "eval_l_values": eval_l_values,
    }, indent=2) + "\n")

    if set(conditions) == {(readout, train_l) for readout in READOUTS for train_l in TRAIN_L_VALUES}:
        write_final_results(args.output_dir, curve_rows, native_rows, eval_l_values, args.seed, args.epoch)


if __name__ == "__main__":
    main()

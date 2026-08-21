"""Run the Easy/Medium/Hard Sudoku cross-evaluation matrix.

Example:
    python cross_evaluate.py \
        --checkpoint easy=checkpoints/easy-run/seed_1/epoch_19.pt \
        --checkpoint medium=checkpoints/medium-run/seed_1/epoch_19.pt \
        --checkpoint hard=checkpoints/hard-run/seed_1/epoch_19.pt
"""

import argparse
import csv
import json
from pathlib import Path

from eval import evaluate_checkpoint


# These match the training bands in config/data/sudoku_{rated_easy,medium,
# rated_hard}.yaml. Bounds are inclusive; None means unbounded.
DIFFICULTY_BOUNDS: dict[str, tuple[int | None, int | None]] = {
    "easy": (0, 15),
    "medium": (16, 30),
    "hard": (31, None),
}


def parse_checkpoint(value: str) -> tuple[str, Path]:
    difficulty, separator, checkpoint = value.partition("=")
    if separator != "=" or difficulty not in DIFFICULTY_BOUNDS or not checkpoint:
        choices = ", ".join(DIFFICULTY_BOUNDS)
        raise argparse.ArgumentTypeError(
            f"checkpoint must be DIFFICULTY=PATH, where DIFFICULTY is one of: {choices}"
        )
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise argparse.ArgumentTypeError(f"checkpoint does not exist: {checkpoint_path}")
    return difficulty, checkpoint_path


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate easy/medium/hard checkpoints on every Sudoku rating band."
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_checkpoint,
        required=True,
        metavar="DIFFICULTY=PATH",
        help="Repeat once each for easy, medium, and hard.",
    )
    parser.add_argument("--split", default="test", help="Dataset split to partition by rating (default: test)")
    parser.add_argument(
        "--eval-dataset-name",
        help="Override the checkpoint's eval dataset. The selected split must contain all three rating bands.",
    )
    parser.add_argument("--output-dir", default="results/cross_evaluation", help="Directory for metrics and per-cell .npz files")
    args = parser.parse_args()

    checkpoints = dict(args.checkpoint)
    missing = set(DIFFICULTY_BOUNDS) - set(checkpoints)
    if missing:
        parser.error(f"missing checkpoint(s): {', '.join(sorted(missing))}")
    if len(checkpoints) != len(args.checkpoint):
        parser.error("each checkpoint difficulty may only be supplied once")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for train_difficulty, checkpoint in checkpoints.items():
        for test_difficulty, (rating_min, rating_max) in DIFFICULTY_BOUNDS.items():
            print(f"\n=== train={train_difficulty}, test={test_difficulty} ===")
            metrics = evaluate_checkpoint(
                str(checkpoint),
                split=args.split,
                eval_rating_min=rating_min,
                eval_rating_max=rating_max,
                eval_dataset_name=args.eval_dataset_name,
                output=str(output_dir / f"train_{train_difficulty}_test_{test_difficulty}.npz"),
            )
            rows.append({
                "train_difficulty": train_difficulty,
                "test_difficulty": test_difficulty,
                "rating_min": rating_min,
                "rating_max": rating_max,
                "checkpoint": str(checkpoint),
                **metrics,
            })

    fieldnames = [
        "train_difficulty", "test_difficulty", "rating_min", "rating_max",
        "checkpoint", "total_samples", "exact_match_accuracy",
    ]
    with (output_dir / "cross_evaluation.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "cross_evaluation.json").open("w") as f:
        json.dump(rows, f, indent=2)

    print("\nExact-match accuracy matrix (rows=train, columns=test)")
    print("             easy      medium      hard")
    for train_difficulty in DIFFICULTY_BOUNDS:
        cells = {
            row["test_difficulty"]: row["exact_match_accuracy"]
            for row in rows if row["train_difficulty"] == train_difficulty
        }
        print(f"{train_difficulty:>6} " + " ".join(f"{cells[test]:10.4%}" for test in DIFFICULTY_BOUNDS))
    print(f"\nSaved detailed results to {output_dir}")


if __name__ == "__main__":
    main()

"""Print blank-cell counts for a local Sudoku-Extreme split."""

import argparse

import numpy as np
from datasets import Features, Value, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Sudoku blank-cell counts in a dataset split.")
    parser.add_argument("--dataset", required=True, help="Local dataset directory, e.g. downloaded-datasets/sudoku-extreme-full")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, split=args.split, features=Features({
        "source": Value("string"),
        "question": Value("string"),
        "answer": Value("string"),
        "rating": Value("int64"),
    }))
    blank_counts = np.fromiter((question.count(".") for question in dataset["question"]), dtype=np.int16)
    bands = {
        "easy (<=53 blanks)": blank_counts <= 53,
        "medium (54-57 blanks)": (blank_counts >= 54) & (blank_counts <= 57),
        "hard (>=58 blanks)": blank_counts >= 58,
    }

    print(f"{args.dataset} [{args.split}]: {len(blank_counts):,} puzzles")
    print(f"blank-count range: {blank_counts.min()}-{blank_counts.max()}")
    for name, mask in bands.items():
        print(f"{name}: {mask.sum():,} ({mask.mean():.2%})")


if __name__ == "__main__":
    main()

"""Print rating quantiles for a local Sudoku-Extreme split."""

import argparse

import numpy as np
from datasets import Features, Value, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize tdoku difficulty ratings in a Sudoku dataset split.")
    parser.add_argument("--dataset", required=True, help="Local dataset directory, e.g. downloaded-datasets/sudoku-extreme-full")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, split=args.split, features=Features({
        "source": Value("string"),
        "question": Value("string"),
        "answer": Value("string"),
        "rating": Value("int64"),
    }))
    ratings = np.asarray(dataset["rating"], dtype=np.int64)
    percentiles = (0, 10, 25, 50, 75, 90, 95, 99, 100)

    print(f"{args.dataset} [{args.split}]: {len(ratings):,} puzzles")
    for percentile, value in zip(percentiles, np.percentile(ratings, percentiles)):
        print(f"p{percentile:>3}: {value:.0f}")


if __name__ == "__main__":
    main()

"""Aggregate Easy-to-Hard rollout evaluations across seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if not paths:
        raise ValueError("at least one result file is required")
    results = [np.load(path) for path in paths]
    cycles = results[0]["rollout_cycles"]
    if any(not np.array_equal(result["rollout_cycles"], cycles) for result in results[1:]):
        raise ValueError("all result files must use the same rollout cycles")
    return cycles, np.stack([result["exact_match"] for result in results])


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot HRM vs RT hard-test accuracy over external rollouts.")
    parser.add_argument("--hrm", nargs="+", required=True, help="HRM eval_result_*.npz files, one per seed")
    parser.add_argument("--rt", nargs="+", required=True, help="RT eval_result_*.npz files, one per seed")
    parser.add_argument("--output", type=Path, required=True, help="Output figure path")
    args = parser.parse_args()

    hrm_cycles, hrm_scores = load_results(args.hrm)
    rt_cycles, rt_scores = load_results(args.rt)
    if not np.array_equal(hrm_cycles, rt_cycles):
        parser.error("HRM and RT results must use the same rollout cycles")

    figure, axis = plt.subplots(figsize=(5.2, 3.4))
    for label, scores, color in (("HRM", hrm_scores, "#3A86FF"), ("RT", rt_scores, "#FB5607")):
        mean = scores.mean(axis=0)
        std = scores.std(axis=0, ddof=1) if len(scores) > 1 else np.zeros_like(mean)
        axis.errorbar(hrm_cycles, mean, yerr=std, label=label, color=color, marker="o", capsize=3)
        print(f"{label}: " + ", ".join(f"rollout {cycle}: {value:.4f} ± {error:.4f}" for cycle, value, error in zip(hrm_cycles, mean, std)))

    axis.set_xscale("log", base=2)
    axis.set_xticks(hrm_cycles, [str(cycle) for cycle in hrm_cycles])
    axis.set_xlabel("External rollout cycles")
    axis.set_ylabel("Sudoku-Extreme exact-match accuracy")
    axis.set_ylim(bottom=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)


if __name__ == "__main__":
    main()

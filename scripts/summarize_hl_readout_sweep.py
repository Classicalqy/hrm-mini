"""Print the exact-match results of the H/L schedule x readout sweep as Markdown."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


SCHEDULES = ("H2L6", "H1L16", "H2L8", "H4L4", "H8L2", "H16L1")
READOUTS = ("h", "l", "hl")


def result_path(checkpoints_root: Path, schedule: str, readout: str, seed: int) -> Path:
    return checkpoints_root / f"{schedule}_{readout}" / f"seed_{seed}" / "eval_result.npz"


def load_exact_match(path: Path) -> float | None:
    if not path.exists():
        return None
    with np.load(path) as result:
        return float(np.mean(result["correctness"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-root", type=Path, default=Path("checkpoints/hl_readout_sweep"))
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    print("| schedule | h | l | hl |")
    print("| --- | ---: | ---: | ---: |")
    for schedule in SCHEDULES:
        cells: list[str] = []
        for readout in READOUTS:
            if schedule == "H2L6" and readout != "h":
                cells.append("—")
                continue
            accuracy = load_exact_match(result_path(args.checkpoints_root, schedule, readout, args.seed))
            cells.append("—" if accuracy is None else f"{accuracy:.4f}")
        print(f"| {schedule} | {' | '.join(cells)} |")


if __name__ == "__main__":
    main()

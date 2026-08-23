import sys
import types
from unittest.mock import patch
import unittest

import torch

# The lightweight local test environment intentionally omits coolname, while
# the server training environment provides it through train.py dependencies.
coolname = types.ModuleType("coolname")
coolname.generate_slug = lambda _count: "stub"
sys.modules.setdefault("coolname", coolname)

from scripts.analyze_long_rollout_msd import RunDirectory
from scripts.core_five_long_rollout import CORE_CONDITIONS, core_runs, fixed_random_samples


def fake_run(condition: str, seed: int) -> RunDirectory:
    return RunDirectory(
        kind="rt" if condition == "RT" else "hrm",
        condition=condition,
        seed=seed,
        directory=None,  # type: ignore[arg-type]
        config=None,  # type: ignore[arg-type]
        l_cycles=None,
        readout="rt" if condition == "RT" else "h",
    )


class CoreFiveProfileTest(unittest.TestCase):
    def test_core_discovery_requires_each_condition_and_seed(self) -> None:
        runs = [fake_run(condition, seed) for condition in CORE_CONDITIONS for seed in (1, 2, 3)]
        with patch("scripts.core_five_long_rollout.discover_hrm_runs", return_value=[run for run in runs if run.kind == "hrm"]), patch(
            "scripts.core_five_long_rollout.discover_rt_runs", return_value=[run for run in runs if run.kind == "rt"],
        ):
            selected = core_runs(None, (1, 2, 3))  # type: ignore[arg-type]
        self.assertEqual([(run.condition, run.seed) for run in selected], [(condition, seed) for condition in CORE_CONDITIONS for seed in (1, 2, 3)])

    def test_core_discovery_reports_missing_seed(self) -> None:
        runs = [fake_run(condition, seed) for condition in CORE_CONDITIONS for seed in (1, 2, 3)]
        runs.pop()
        with patch("scripts.core_five_long_rollout.discover_hrm_runs", return_value=[run for run in runs if run.kind == "hrm"]), patch(
            "scripts.core_five_long_rollout.discover_rt_runs", return_value=[run for run in runs if run.kind == "rt"],
        ):
            with self.assertRaisesRegex(FileNotFoundError, "RT/seed_3"):
                core_runs(None, (1, 2, 3))  # type: ignore[arg-type]

    def test_fixed_random_samples_are_repeatable(self) -> None:
        loader = [(torch.arange(20).reshape(10, 2), torch.zeros(10, 2, dtype=torch.long))]
        first, first_indices = fixed_random_samples(loader, samples=4, seed=9)
        second, second_indices = fixed_random_samples(loader, samples=4, seed=9)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first_indices.tolist(), second_indices.tolist())


if __name__ == "__main__":
    unittest.main()

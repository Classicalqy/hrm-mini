import sys
import types
import unittest

import torch

coolname = types.ModuleType("coolname")
coolname.generate_slug = lambda _count: "stub"
sys.modules.setdefault("coolname", coolname)

from arch.hrm import HRM
from scripts.analyze_long_rollout_msd import RunDirectory
from scripts.core_h_l_clock_dynamics import CORE_KEYS, H_BOUNDARIES, H_LAGS, TOTAL_H_UPDATES, Unit, advance_block, units
from scripts.core_five_l_depth_long_rollout import initial_hrm_state
from test_hrm_readout import tiny_config


def fake_run(condition: str, seed: int) -> RunDirectory:
    return RunDirectory("rt" if condition == "RT" else "hrm", condition, seed, None, None, None if condition == "RT" else 6, "rt" if condition == "RT" else "h")  # type: ignore[arg-type]


class CoreHLClockTest(unittest.TestCase):
    def test_protocol_has_no_burnin_and_four_equal_h_segments(self) -> None:
        self.assertEqual(TOTAL_H_UPDATES, 160)
        self.assertEqual(H_BOUNDARIES.tolist(), [0, 40, 80, 120, 160])
        self.assertEqual(H_LAGS.tolist(), [1, 2, 3, 4, 6, 8, 12, 16])

    def test_five_conditions_make_fifteen_seed_units(self) -> None:
        names = {name for name, _ in CORE_KEYS}
        runs = [fake_run(name, seed) for name in names for seed in (1, 2, 3)]
        selected = units(runs, (1, 2, 3))
        self.assertEqual(len(selected), 15)
        self.assertEqual({(item.run.condition, item.eval_l) for item in selected}, set(CORE_KEYS))

    def test_one_manual_h_block_matches_one_cycle_hrm_forward(self) -> None:
        torch.manual_seed(4)
        model = HRM(tiny_config("h", h_cycles=1, l_cycles=6))
        x = torch.randint(0, 11, (2, 4))
        native, logits = model(model.initial_carry, x)
        with torch.inference_mode():
            state, _ = advance_block(model, initial_hrm_state(model, x), model.embed(x), 6)
            manual_logits = model.readout_logits(*state)
        self.assertTrue(torch.equal(state[0], native["z_H"]))
        self.assertTrue(torch.equal(state[1], native["z_L"]))
        self.assertTrue(torch.equal(manual_logits, logits))


if __name__ == "__main__":
    unittest.main()

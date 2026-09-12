import sys
import types
import unittest

import torch

# train.py imports coolname, which the lightweight local test environment omits.
coolname = types.ModuleType("coolname")
coolname.generate_slug = lambda _count: "stub"
sys.modules.setdefault("coolname", coolname)

from arch.hrm import HRM
from scripts.evaluate_h2l6_h_interventions import advance_h_block
from test_hrm_readout import tiny_config


class HInterventionTest(unittest.TestCase):
    def test_unmodified_h_block_matches_native_single_h_forward(self) -> None:
        torch.manual_seed(11)
        model = HRM(tiny_config("h", h_cycles=1, l_cycles=3))
        input_ids = torch.randint(0, 11, (2, 4))
        expected_carry, expected_logits = model(model.initial_carry, input_ids)
        actual_carry, actual_logits = advance_h_block(model, model.initial_carry, input_ids)
        self.assertTrue(torch.equal(expected_logits, actual_logits))
        self.assertTrue(torch.equal(expected_carry["z_H"], actual_carry["z_H"]))
        self.assertTrue(torch.equal(expected_carry["z_L"], actual_carry["z_L"]))

    def test_freeze_h_preserves_the_specified_state(self) -> None:
        torch.manual_seed(12)
        model = HRM(tiny_config("l", h_cycles=1, l_cycles=2))
        input_ids = torch.randint(0, 11, (2, 4))
        frozen = model.initial_carry["z_H"]
        carry, _ = advance_h_block(model, model.initial_carry, input_ids, frozen_h=frozen)
        self.assertTrue(torch.equal(carry["z_H"], frozen))
        self.assertFalse(torch.equal(carry["z_L"], model.initial_carry["z_L"]))


if __name__ == "__main__":
    unittest.main()

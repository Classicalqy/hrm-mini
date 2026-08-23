import unittest

import torch

from arch.hrm import HRM

from test_hrm_readout import CountingIdentity, tiny_config


class HRMTraceTest(unittest.TestCase):
    def test_trace_matches_regular_forward_and_update_counts(self) -> None:
        torch.manual_seed(0)
        model = HRM(tiny_config("hl", h_cycles=2, l_cycles=3))
        input_ids = torch.randint(0, 11, (2, 4))
        carry = {key: value.clone() for key, value in model.initial_carry.items()}
        regular_carry, regular_logits = model(carry, input_ids)

        events: list[str] = []
        traced_carry, traced_logits = model.forward_with_trace(
            {key: value.clone() for key, value in model.initial_carry.items()},
            input_ids,
            lambda event, _z_h, _z_l: events.append(event),
        )

        self.assertEqual(events, ["l", "l", "l", "h", "l", "l", "l", "h"])
        self.assertTrue(torch.equal(regular_logits, traced_logits))
        self.assertTrue(torch.equal(regular_carry["z_H"], traced_carry["z_H"]))
        self.assertTrue(torch.equal(regular_carry["z_L"], traced_carry["z_L"]))

    def test_hl_logit_terms_recompose_the_full_head(self) -> None:
        model = HRM(tiny_config("hl"))
        z_h = torch.randn(2, 4, 8)
        z_l = torch.randn(2, 4, 8)
        h_logits, l_logits = model.split_hl_readout_logits(z_h, z_l)
        self.assertTrue(torch.allclose(h_logits + l_logits, model.readout_logits(z_h, z_l), atol=1e-6, rtol=1e-5))

    def test_shared_initial_h_state_broadcasts_for_intermediate_readout(self) -> None:
        model = HRM(tiny_config("hl"))
        z_h = torch.randn(8)
        z_l = torch.randn(2, 4, 8)
        logits = model.readout_logits(z_h, z_l)
        h_logits, l_logits = model.split_hl_readout_logits(z_h, z_l)
        self.assertEqual(logits.shape, (2, 4, 11))
        self.assertTrue(torch.allclose(h_logits + l_logits, logits))

    def test_trace_calls_levels_expected_number_of_times(self) -> None:
        model = HRM(tiny_config("h", h_cycles=4, l_cycles=2))
        h_level = CountingIdentity()
        l_level = CountingIdentity()
        model.H_level = h_level
        model.L_level = l_level
        input_ids = torch.randint(0, 11, (2, 4))

        model.forward_with_trace(model.initial_carry, input_ids, lambda *_args: None)

        self.assertEqual(h_level.calls, 4)
        self.assertEqual(l_level.calls, 8)

    def test_h_only_trace_preserves_forward_and_suppresses_l_callbacks(self) -> None:
        model = HRM(tiny_config("h", h_cycles=2, l_cycles=3))
        input_ids = torch.randint(0, 11, (2, 4))
        regular_carry, regular_logits = model(model.initial_carry, input_ids)
        events: list[str] = []
        traced_carry, traced_logits = model.forward_with_trace(
            model.initial_carry,
            input_ids,
            lambda event, _z_h, _z_l: events.append(event),
            events=("h",),
        )
        self.assertEqual(events, ["h", "h"])
        self.assertTrue(torch.equal(regular_logits, traced_logits))
        self.assertTrue(torch.equal(regular_carry["z_H"], traced_carry["z_H"]))
        self.assertTrue(torch.equal(regular_carry["z_L"], traced_carry["z_L"]))

    def test_two_single_h_blocks_match_native_h2_rollout(self) -> None:
        torch.manual_seed(7)
        model = HRM(tiny_config("h", h_cycles=2, l_cycles=3))
        input_ids = torch.randint(0, 11, (2, 4))
        native_carry, native_logits = model(model.initial_carry, input_ids)

        model.H_cycles = 1
        carry, _ = model(model.initial_carry, input_ids)
        split_carry, split_logits = model(carry, input_ids)

        self.assertTrue(torch.equal(native_logits, split_logits))
        self.assertTrue(torch.equal(native_carry["z_H"], split_carry["z_H"]))
        self.assertTrue(torch.equal(native_carry["z_L"], split_carry["z_L"]))


if __name__ == "__main__":
    unittest.main()

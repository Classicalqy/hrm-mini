import unittest

import torch
from torch import nn

from arch.hrm import HRM


def tiny_config(readout: str = "h", h_cycles: int = 2, l_cycles: int = 3) -> dict[str, object]:
    return {
        "vocab_size": 11,
        "seq_len": 4,
        "num_layers": 1,
        "hidden_size": 8,
        "intermediate_size": 16,
        "head_dim": 4,
        "is_causal": False,
        "norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "H_cycles": h_cycles,
        "L_cycles": l_cycles,
        "bptt": True,
        "readout": readout,
        "forward_dtype": "float32",
    }


class CountingIdentity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return x


class RecordingHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.input = x
        return x


class HRMReadoutTest(unittest.TestCase):
    def test_readout_shapes_and_head_widths(self) -> None:
        input_ids = torch.randint(0, 11, (2, 4))
        for readout, expected_width in (("h", 8), ("l", 8), ("hl", 16)):
            with self.subTest(readout=readout):
                model = HRM(tiny_config(readout))
                _, logits = model(model.initial_carry, input_ids)
                self.assertEqual(logits.shape, (2, 4, 11))
                self.assertEqual(model.lm_head.in_features, expected_width)

    def test_default_h_readout_preserves_head_shape(self) -> None:
        model = HRM(tiny_config())
        self.assertEqual(model.readout, "h")
        self.assertEqual(model.lm_head.weight.shape, (11, 8))

    def test_schedule_performs_expected_number_of_updates(self) -> None:
        input_ids = torch.randint(0, 11, (2, 4))
        for h_cycles, l_cycles in ((1, 16), (2, 8), (4, 4), (8, 2), (16, 1)):
            with self.subTest(h_cycles=h_cycles, l_cycles=l_cycles):
                model = HRM(tiny_config(h_cycles=h_cycles, l_cycles=l_cycles))
                h_level = CountingIdentity()
                l_level = CountingIdentity()
                model.H_level = h_level
                model.L_level = l_level
                model.lm_head = RecordingHead()

                model(model.initial_carry, input_ids)

                self.assertEqual(h_level.calls, h_cycles)
                self.assertEqual(l_level.calls, h_cycles * l_cycles)

    def test_hl_readout_concatenates_states(self) -> None:
        model = HRM(tiny_config("hl"))
        head = RecordingHead()
        model.lm_head = head
        input_ids = torch.randint(0, 11, (2, 4))

        model(model.initial_carry, input_ids)

        self.assertIsNotNone(head.input)
        assert head.input is not None
        self.assertEqual(head.input.shape[-1], 16)

    def test_invalid_readout_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HRM(tiny_config("invalid"))


if __name__ == "__main__":
    unittest.main()

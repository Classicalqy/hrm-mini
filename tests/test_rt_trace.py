import unittest

import torch

from arch.rt import RecurrentTransformer
from test_hrm_readout import tiny_config


class RTTraceTest(unittest.TestCase):
    def test_trace_matches_regular_forward(self) -> None:
        config = tiny_config()
        config.pop("H_cycles")
        config.pop("L_cycles")
        config.pop("readout")
        config["cycles"] = 3
        model = RecurrentTransformer(config)
        input_ids = torch.randint(0, 11, (2, 4))
        regular_carry, regular_logits = model(model.initial_carry, input_ids)
        events: list[str] = []
        traced_carry, traced_logits = model.forward_with_trace(
            model.initial_carry,
            input_ids,
            lambda event, _z: events.append(event),
        )
        self.assertEqual(events, ["z", "z", "z"])
        self.assertTrue(torch.equal(regular_logits, traced_logits))
        self.assertTrue(torch.equal(regular_carry["z"], traced_carry["z"]))


if __name__ == "__main__":
    unittest.main()

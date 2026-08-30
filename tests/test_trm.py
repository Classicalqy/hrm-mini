import unittest

import torch

from arch.trm import TRM


def _config(*, bptt: bool) -> dict[str, object]:
    return {
        "vocab_size": 10,
        "seq_len": 6,
        "is_causal": False,
        "num_layers": 1,
        "hidden_size": 16,
        "intermediate_size": 32,
        "head_dim": 8,
        "norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "H_cycles": 2,
        "L_cycles": 3,
        "bptt": bptt,
        "forward_dtype": "float32",
    }


class TRMTests(unittest.TestCase):
    def test_model_has_one_shared_core_and_standard_interface(self):
        model = TRM(_config(bptt=True))
        self.assertTrue(hasattr(model, "core"))
        self.assertFalse(hasattr(model, "H_level"))
        self.assertFalse(hasattr(model, "L_level"))
        self.assertEqual(len([name for name, _ in model.named_modules() if name == "core"]), 1)

        carry, logits = model(model.initial_carry, torch.randint(0, 10, (2, 6)))
        self.assertEqual(set(carry), {"z_H", "z_L"})
        self.assertEqual(tuple(carry["z_H"].shape), (2, 6, 16))
        self.assertEqual(tuple(carry["z_L"].shape), (2, 6, 16))
        self.assertFalse(carry["z_H"].requires_grad)
        self.assertEqual(tuple(logits.shape), (2, 6, 10))

    def test_bptt_controls_recursive_gradient_history(self):
        inputs = torch.randint(0, 10, (2, 6))
        expected_calls = 2 * (3 + 1)

        for bptt, expected_grad_calls in ((True, expected_calls), (False, 2)):
            model = TRM(_config(bptt=bptt))
            grad_enabled_calls: list[bool] = []
            handle = model.core.register_forward_hook(
                lambda _module, _args, _output: grad_enabled_calls.append(torch.is_grad_enabled())
            )
            _, logits = model(model.initial_carry, inputs)
            logits.sum().backward()
            handle.remove()

            self.assertEqual(len(grad_enabled_calls), expected_calls)
            self.assertEqual(sum(grad_enabled_calls), expected_grad_calls)
            self.assertIsNotNone(model.core.layers[0].mlp.up_proj.weight.grad)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import yaml

from arch.rt import RecurrentTransformer
from .common import RolloutOutput, StepOutput


class NativeRTAdapter(nn.Module):
    """Micro-step tracing adapter for an unmodified native RT checkpoint."""

    def __init__(self, model: RecurrentTransformer, cycles: int) -> None:
        super().__init__()
        self.model = model
        self.cycles = int(cycles)
        self.hidden_size = int(model.z_init.shape[-1])
        self.vocab_size = int(model.lm_head.weight.shape[0])

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, device: torch.device | str = "cpu") -> "NativeRTAdapter":
        checkpoint = Path(checkpoint)
        config_path = checkpoint.parent / "model_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint config: {config_path}")
        with config_path.open() as handle:
            config = yaml.safe_load(handle)
        arch: dict[str, Any] = dict(config["arch"])
        arch.pop("name", None)
        arch |= {"vocab_size": 10, "seq_len": 82, "is_causal": False}
        model = RecurrentTransformer(arch).to(device)
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict({key.removeprefix("_orig_mod."): value for key, value in state.items()})
        model.eval()
        return cls(model, cycles=int(arch["cycles"]))

    def initial_state(self, input_ids: Tensor) -> Tensor:
        return self.model.z_init.view(1, 1, -1).expand(input_ids.shape[0], input_ids.shape[1], -1)

    def decode(self, state: Tensor) -> Tensor:
        return self.model.lm_head(state)

    def step(self, state: Tensor, input_ids: Tensor, step_index: int, total_steps: int) -> StepOutput:
        del step_index, total_steps
        next_state = self.model.core(state + self.model.embed(input_ids))
        return StepOutput(state=next_state, logits=self.decode(next_state), update=next_state - state)

    def rollout(self, input_ids: Tensor, steps: int, return_trace: bool = False) -> RolloutOutput:
        state = self.initial_state(input_ids)
        states = [state] if return_trace else None
        logits = self.decode(state)
        logits_trace = [logits] if return_trace else None
        updates: list[Tensor] | None = [] if return_trace else None
        for index in range(steps):
            output = self.step(state, input_ids, index, steps)
            state, logits = output.state, output.logits
            if return_trace:
                assert states is not None and logits_trace is not None and updates is not None
                states.append(state); logits_trace.append(logits); updates.append(output.update)
        return RolloutOutput(state, logits, states, logits_trace, updates)

    @torch.inference_mode()
    def validate_group_parity(self, input_ids: Tensor, atol: float = 2e-3, rtol: float = 2e-3) -> None:
        native_carry, native_logits = self.model({"z": self.initial_state(input_ids)}, input_ids)
        traced = self.rollout(input_ids, self.cycles)
        torch.testing.assert_close(traced.state, native_carry["z"], atol=atol, rtol=rtol)
        torch.testing.assert_close(traced.logits, native_logits, atol=atol, rtol=rtol)

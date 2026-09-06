from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F

from .common import MatchedDynamicsBase, StepOutput


class MatchedRecurrentTransformer(MatchedDynamicsBase):
    """Discrete recurrent control with the exact same trainable backbone as Flow."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

    def step(self, state: Tensor, input_ids: Tensor, step_index: int, total_steps: int) -> StepOutput:
        t = torch.full(
            (input_ids.shape[0],),
            step_index / max(total_steps - 1, 1),
            device=input_ids.device,
            dtype=torch.float32,
        )
        next_state = self.core(state + self.condition(input_ids, t))
        update = next_state - state
        return StepOutput(state=next_state, logits=self.decode(next_state), update=update)

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor,
        state: Tensor,
        start_step: int,
        total_steps: int,
        micro_steps: int,
        curvature_lambda: float = 0.0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        updates: list[Tensor] = []
        logits = self.decode(state)
        for offset in range(micro_steps):
            output = self.step(state, input_ids, start_step + offset, total_steps)
            state, logits = output.state, output.logits
            updates.append(output.update)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1).long())
        curvature = torch.zeros((), device=state.device, dtype=torch.float32)
        if len(updates) > 1:
            directions = [F.normalize(update.to(torch.float32), dim=-1, eps=1e-6) for update in updates]
            curvature = torch.stack([
                (directions[index + 1] - directions[index]).square().mean()
                for index in range(len(directions) - 1)
            ]).mean()
        loss = ce + float(curvature_lambda) * curvature
        return loss, ce.detach(), curvature.detach(), state.detach(), logits.detach()

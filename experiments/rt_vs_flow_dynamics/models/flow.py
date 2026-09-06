from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F

from .common import MatchedDynamicsBase, StepOutput


def orthogonal_bend(delta: Tensor) -> Tensor:
    """Deterministic bend with the same norm as delta and orthogonal to it."""
    raw = torch.roll(delta.to(torch.float32), shifts=1, dims=-1)
    denominator = delta.to(torch.float32).square().sum(dim=-1, keepdim=True).clamp_min(1e-12)
    bend = raw - (raw * delta).sum(dim=-1, keepdim=True) / denominator * delta
    bend = F.normalize(bend, dim=-1, eps=1e-6) * denominator.sqrt()
    return bend.to(delta.dtype)


def interpolant(z0: Tensor, z1: Tensor, t: Tensor, beta: float = 0.0) -> tuple[Tensor, Tensor]:
    """Return gamma(t) and its analytic time derivative."""
    while t.ndim < z0.ndim:
        t = t.unsqueeze(-1)
    t = t.to(dtype=z0.dtype)
    delta = z1 - z0
    bend = orthogonal_bend(delta)
    path = (1.0 - t) * z0 + t * z1 + 4.0 * float(beta) * t * (1.0 - t) * bend
    velocity = delta + 4.0 * float(beta) * (1.0 - 2.0 * t) * bend
    return path, velocity


class FlowMatchingTransformer(MatchedDynamicsBase):
    """Time-conditioned conditional flow matching model with Euler inference."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

    def velocity(self, state: Tensor, input_ids: Tensor, t: Tensor) -> Tensor:
        return self.core(state + self.condition(input_ids, t))

    def step(self, state: Tensor, input_ids: Tensor, step_index: int, total_steps: int) -> StepOutput:
        t = torch.full(
            (input_ids.shape[0],), step_index / total_steps, device=input_ids.device, dtype=torch.float32
        )
        velocity = self.velocity(state, input_ids, t)
        update = velocity / float(total_steps)
        next_state = state + update
        return StepOutput(
            state=next_state,
            logits=self.decode(next_state),
            update=update,
            velocity=velocity,
        )

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor,
        times: Tensor,
        beta: float = 0.0,
        fm_weight: float = 1.0,
        ce_weight: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        z0 = self.initial_state(input_ids)
        z1 = self.target_state(targets)
        fm_terms: list[Tensor] = []
        ce_terms: list[Tensor] = []
        for time in times:
            path, target_velocity = interpolant(z0, z1, time, beta=beta)
            predicted_velocity = self.velocity(path, input_ids, time)
            fm_terms.append(F.mse_loss(predicted_velocity.to(torch.float32), target_velocity.to(torch.float32)))
            shape = (time.shape[0],) + (1,) * (path.ndim - 1)
            remaining = 1.0 - time.view(shape).to(path.dtype)
            predicted_endpoint = path + remaining * predicted_velocity
            logits = self.decode(predicted_endpoint)
            ce_terms.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1).long()))
        fm = torch.stack(fm_terms).mean()
        ce = torch.stack(ce_terms).mean()
        return float(fm_weight) * fm + float(ce_weight) * ce, fm.detach(), ce.detach()

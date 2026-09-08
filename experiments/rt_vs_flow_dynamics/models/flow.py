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


def relative_velocity_loss(predicted: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    """Dimension-independent vector regression loss.

    A zero prediction has loss approximately one instead of ``1 / hidden_size``.
    """
    numerator = (predicted.to(torch.float32) - target.to(torch.float32)).square().sum(dim=-1)
    denominator = target.to(torch.float32).square().sum(dim=-1).detach().clamp_min(eps)
    return (numerator / denominator).mean()


def reconstruct_endpoint(
    path: Tensor,
    predicted_velocity: Tensor,
    z0: Tensor,
    z1: Tensor,
    t: Tensor,
    beta: float = 0.0,
) -> Tensor:
    """Recover the endpoint from a tangent prediction on the analytic path."""
    while t.ndim < path.ndim:
        t = t.unsqueeze(-1)
    remaining = 1.0 - t.to(path.dtype)
    correction = 4.0 * float(beta) * remaining.square() * orthogonal_bend(z1 - z0)
    return path + remaining * predicted_velocity - correction


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

    def forward_v1(
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
            predicted_endpoint = reconstruct_endpoint(path, predicted_velocity, z0, z1, time, beta)
            logits = self.decode(predicted_endpoint)
            ce_terms.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1).long()))
        fm = torch.stack(fm_terms).mean()
        ce = torch.stack(ce_terms).mean()
        return float(fm_weight) * fm + float(ce_weight) * ce, fm.detach(), ce.detach()

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor,
        state: Tensor | None = None,
        start_step: int = 0,
        total_steps: int = 1,
        micro_steps: int = 1,
        beta: float = 0.0,
        on_policy_ratio: float = 0.0,
        fm_weight: float = 1.0,
        endpoint_ce_weight: float = 1.0,
        endpoint_cos_weight: float = 1.0,
        legacy_times: Tensor | None = None,
        legacy_ce_weight: float = 1.0,
    ) -> tuple[Tensor, dict[str, Tensor], Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """Train one seven-call Flow chunk while carrying its generated state.

        ``on_policy_ratio=0`` is corrected teacher-path CFM.  Increasing the
        ratio moves the field inputs onto the model-generated rollout without
        spending additional backbone calls.
        """
        if legacy_times is not None:
            return self.forward_v1(
                input_ids, targets, legacy_times, beta=beta,
                fm_weight=fm_weight, ce_weight=legacy_ce_weight,
            )
        if state is None:
            raise ValueError("state is required for Flow V2 chunk training")
        if not 0.0 <= on_policy_ratio <= 1.0:
            raise ValueError("on_policy_ratio must be in [0, 1]")
        z0 = self.initial_state(input_ids)
        z1 = self.target_state(targets)
        relative_terms: list[Tensor] = []
        cosine_terms: list[Tensor] = []
        predicted_norms: list[Tensor] = []
        target_norms: list[Tensor] = []
        endpoint = state
        carry = state
        for offset in range(micro_steps):
            index = start_step + offset
            time = torch.full(
                (input_ids.shape[0],), index / float(total_steps),
                device=input_ids.device, dtype=torch.float32,
            )
            teacher_state, target_velocity = interpolant(z0, z1, time, beta=beta)
            field_state = torch.lerp(teacher_state, carry, float(on_policy_ratio))
            predicted_velocity = self.velocity(field_state, input_ids, time)
            relative_terms.append(relative_velocity_loss(predicted_velocity, target_velocity))
            cosine_terms.append(
                F.cosine_similarity(
                    predicted_velocity.to(torch.float32), target_velocity.to(torch.float32), dim=-1, eps=1e-6
                ).mean()
            )
            predicted_norms.append(predicted_velocity.to(torch.float32).norm(dim=-1).mean())
            target_norms.append(target_velocity.to(torch.float32).norm(dim=-1).mean())
            endpoint = reconstruct_endpoint(
                field_state, predicted_velocity, z0, z1, time, beta=beta
            )
            carry = carry + predicted_velocity / float(total_steps)

        at_final_chunk = start_step + micro_steps >= total_steps
        supervised_state = carry if at_final_chunk else endpoint
        supervised_logits = self.decode(supervised_state)
        endpoint_ce = F.cross_entropy(
            supervised_logits.reshape(-1, supervised_logits.shape[-1]), targets.reshape(-1).long()
        )
        endpoint_cos = (
            1.0
            - F.cosine_similarity(
                supervised_state.to(torch.float32), z1.to(torch.float32), dim=-1, eps=1e-6
            )
        ).mean()
        relative = torch.stack(relative_terms).mean()
        loss = (
            float(fm_weight) * relative
            + float(endpoint_ce_weight) * endpoint_ce
            + float(endpoint_cos_weight) * endpoint_cos
        )
        predictions = supervised_logits.argmax(dim=-1)
        metrics = {
            "relative_velocity": relative.detach(),
            "velocity_cosine": torch.stack(cosine_terms).mean().detach(),
            "predicted_velocity_norm": torch.stack(predicted_norms).mean().detach(),
            "target_velocity_norm": torch.stack(target_norms).mean().detach(),
            "endpoint_ce": endpoint_ce.detach(),
            "endpoint_cosine_loss": endpoint_cos.detach(),
            "rollout_cell_accuracy": (predictions == targets).to(torch.float32).mean().detach(),
            "rollout_exact_match": (predictions == targets).all(dim=-1).to(torch.float32).mean().detach(),
            "on_policy_ratio": torch.tensor(float(on_policy_ratio), device=state.device),
        }
        return loss, metrics, carry.detach(), self.decode(carry).detach()

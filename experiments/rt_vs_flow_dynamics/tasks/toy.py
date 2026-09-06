from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


FAMILY_NAMES = ("straight", "semicircle", "sine_1", "sine_2", "sine_4")


@dataclass
class ToyBatch:
    context: Tensor
    path: Tensor
    velocity: Tensor
    family: Tensor


def generate_toy_trajectories(count: int, steps: int, seed: int) -> ToyBatch:
    if count <= 0 or steps < 2:
        raise ValueError("count must be positive and steps must be >= 2")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    start = 2.0 * torch.rand(count, 2, generator=generator) - 1.0
    angle = 2.0 * math.pi * torch.rand(count, generator=generator)
    length = 0.75 + 1.25 * torch.rand(count, generator=generator)
    direction = torch.stack((angle.cos(), angle.sin()), dim=-1)
    perpendicular = torch.stack((-direction[:, 1], direction[:, 0]), dim=-1)
    delta = length[:, None] * direction
    end = start + delta
    family = torch.randint(len(FAMILY_NAMES), (count,), generator=generator)
    times = torch.linspace(0.0, 1.0, steps)
    t = times.view(1, steps, 1)
    path = start[:, None, :] + t * delta[:, None, :]
    velocity = delta[:, None, :].expand(-1, steps, -1).clone()

    semicircle = family == 1
    if semicircle.any():
        ts = times.view(1, steps, 1)
        radius = length[semicircle, None, None] / 2.0
        midpoint = (start[semicircle] + end[semicircle])[:, None, :] / 2.0
        e = direction[semicircle, None, :]
        p = perpendicular[semicircle, None, :]
        path[semicircle] = midpoint + radius * (
            -torch.cos(math.pi * ts) * e + torch.sin(math.pi * ts) * p
        )
        velocity[semicircle] = radius * math.pi * (
            torch.sin(math.pi * ts) * e + torch.cos(math.pi * ts) * p
        )

    for family_index, frequency in ((2, 1), (3, 2), (4, 4)):
        mask = family == family_index
        if not mask.any():
            continue
        amplitude = 0.3 * length[mask, None, None]
        phase = frequency * math.pi * t
        path[mask] += amplitude * torch.sin(phase) * perpendicular[mask, None, :]
        velocity[mask] += (
            amplitude * frequency * math.pi * torch.cos(phase) * perpendicular[mask, None, :]
        )

    context = torch.cat((start, end, F.one_hot(family, len(FAMILY_NAMES)).float()), dim=-1)
    return ToyBatch(context=context, path=path, velocity=velocity, family=family)


def toy_time_features(t: Tensor) -> Tensor:
    t = t.to(torch.float32)
    return torch.stack(
        (torch.sin(2 * math.pi * t), torch.cos(2 * math.pi * t),
         torch.sin(4 * math.pi * t), torch.cos(4 * math.pi * t)),
        dim=-1,
    )


class ToyField(nn.Module):
    def __init__(self, context_size: int = 9, width: int = 128, depth: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        size = 2 + context_size + 4
        for _ in range(depth):
            layers.extend((nn.Linear(size, width), nn.SiLU()))
            size = width
        layers.append(nn.Linear(size, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, state: Tensor, context: Tensor, t: Tensor) -> Tensor:
        return self.network(torch.cat((state, context, toy_time_features(t)), dim=-1))


class ToyRT(nn.Module):
    def __init__(self, width: int = 128, depth: int = 3) -> None:
        super().__init__()
        self.field = ToyField(width=width, depth=depth)

    def rollout(self, context: Tensor, steps: int) -> Tensor:
        state = context[:, :2]
        states = [state]
        for index in range(steps - 1):
            t = torch.full((state.shape[0],), index / (steps - 1), device=state.device)
            state = self.field(state, context, t)
            states.append(state)
        return torch.stack(states, dim=1)


class ToyFlow(nn.Module):
    def __init__(self, width: int = 128, depth: int = 3) -> None:
        super().__init__()
        self.field = ToyField(width=width, depth=depth)

    def rollout(self, context: Tensor, steps: int) -> Tensor:
        state = context[:, :2]
        states = [state]
        dt = 1.0 / (steps - 1)
        for index in range(steps - 1):
            t = torch.full((state.shape[0],), index / (steps - 1), device=state.device)
            state = state + dt * self.field(state, context, t)
            states.append(state)
        return torch.stack(states, dim=1)


def toy_path_metrics(predicted: Tensor, target: Tensor) -> dict[str, Tensor]:
    delta = torch.diff(predicted, dim=1)
    target_delta = torch.diff(target, dim=1)
    length = delta.norm(dim=-1).sum(dim=1)
    displacement = (predicted[:, -1] - predicted[:, 0]).norm(dim=-1).clamp_min(1e-8)
    directions = F.normalize(delta, dim=-1, eps=1e-8)
    cosines = (directions[:, 1:] * directions[:, :-1]).sum(dim=-1).clamp(-1, 1)
    target_direction = F.normalize(target[:, -1] - target[:, 0], dim=-1, eps=1e-8)
    progress = (delta * target_direction[:, None, :]).sum(dim=-1)
    covariance = delta.transpose(1, 2) @ delta
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    effective_rank = eigenvalues.sum(dim=-1).square() / eigenvalues.square().sum(dim=-1).clamp_min(1e-12)
    return {
        "endpoint_error": (predicted[:, -1] - target[:, -1]).norm(dim=-1),
        "path_rmse": (predicted - target).square().mean(dim=(1, 2)).sqrt(),
        "tortuosity": length / displacement,
        "mean_turn_angle": torch.acos(cosines).mean(dim=1),
        "direction_autocorrelation": cosines.mean(dim=1),
        "backtrack_rate": (progress < 0).float().mean(dim=1),
        "target_step_rmse": (delta - target_delta).square().mean(dim=(1, 2)).sqrt(),
        "effective_rank": effective_rank,
    }

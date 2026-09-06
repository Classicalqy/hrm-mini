from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from arch.layers import CastedScaledEmbedding, Transformer, TransformerConfig, trunc_normal_init_


@dataclass
class StepOutput:
    state: Tensor
    logits: Tensor
    update: Tensor
    velocity: Tensor | None = None


@dataclass
class RolloutOutput:
    state: Tensor
    logits: Tensor
    states: list[Tensor] | None = None
    logits_trace: list[Tensor] | None = None
    updates: list[Tensor] | None = None


def sinusoidal_time_embedding(t: Tensor, hidden_size: int) -> Tensor:
    """Parameter-free time embedding with output shape ``t.shape + [hidden_size]``."""
    if hidden_size < 2:
        raise ValueError("hidden_size must be at least two")
    half = hidden_size // 2
    frequencies = torch.exp(
        torch.linspace(0.0, -math.log(10_000.0), half, device=t.device, dtype=torch.float32)
    )
    angles = t.to(torch.float32).unsqueeze(-1) * frequencies * (2.0 * math.pi)
    result = torch.cat((angles.sin(), angles.cos()), dim=-1)
    if result.shape[-1] < hidden_size:
        result = F.pad(result, (0, hidden_size - result.shape[-1]))
    return result


def make_orthogonal_codebook(vocab_size: int, hidden_size: int, seed: int) -> Tensor:
    if hidden_size < vocab_size:
        raise ValueError("hidden_size must be >= vocab_size for an orthogonal codebook")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn(hidden_size, vocab_size, generator=generator, dtype=torch.float32)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q.T.contiguous()


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


class FixedCodebookDecoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, seed: int, temperature: float = 10.0) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.codebook = nn.Buffer(make_orthogonal_codebook(vocab_size, hidden_size, seed), persistent=True)

    def encode(self, token_ids: Tensor) -> Tensor:
        return F.embedding(token_ids.long(), self.codebook)

    def forward(self, state: Tensor) -> Tensor:
        normalized = F.normalize(state.to(torch.float32), dim=-1)
        return self.temperature * normalized @ self.codebook.T


class MatchedDynamicsBase(nn.Module):
    """Shared parameterization used by the controlled RT and Flow models."""

    backbone_calls_per_step = 1

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        transformer_config = TransformerConfig(**{
            key: config[key]
            for key in (
                "seq_len", "num_layers", "hidden_size", "intermediate_size", "head_dim",
                "is_causal", "norm_eps", "rope_theta",
            )
        })
        self.hidden_size = transformer_config.hidden_size
        self.vocab_size = int(config["vocab_size"])
        self.forward_dtype = getattr(torch, str(config.get("forward_dtype", "float32")))
        self.core = Transformer(transformer_config)
        self.embed = CastedScaledEmbedding(self.vocab_size, self.hidden_size, cast_to=self.forward_dtype)
        self.decoder = FixedCodebookDecoder(
            self.vocab_size,
            self.hidden_size,
            seed=int(config.get("codebook_seed", 20260906)),
            temperature=float(config.get("decoder_temperature", 10.0)),
        )
        self.z_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.hidden_size, dtype=self.forward_dtype)), persistent=True
        )

    def initial_state(self, input_ids: Tensor) -> Tensor:
        return self.z_init.view(1, 1, -1).expand(input_ids.shape[0], input_ids.shape[1], -1)

    def condition(self, input_ids: Tensor, t: Tensor) -> Tensor:
        x = self.embed(input_ids)
        time = sinusoidal_time_embedding(t, self.hidden_size).to(dtype=x.dtype)
        while time.ndim < x.ndim:
            time = time.unsqueeze(-2)
        return x + time

    def decode(self, state: Tensor) -> Tensor:
        return self.decoder(state)

    def target_state(self, targets: Tensor) -> Tensor:
        return self.decoder.encode(targets).to(dtype=self.forward_dtype)

    def rollout(self, input_ids: Tensor, steps: int, return_trace: bool = False) -> RolloutOutput:
        if steps <= 0:
            raise ValueError("steps must be positive")
        state = self.initial_state(input_ids)
        states = [state] if return_trace else None
        initial_logits = self.decode(state)
        logits_trace = [initial_logits] if return_trace else None
        updates: list[Tensor] | None = [] if return_trace else None
        logits = initial_logits
        for index in range(steps):
            output = self.step(state, input_ids, index, steps)
            state, logits = output.state, output.logits
            if return_trace:
                assert states is not None and logits_trace is not None and updates is not None
                states.append(state)
                logits_trace.append(logits)
                updates.append(output.update)
        return RolloutOutput(state=state, logits=logits, states=states, logits_trace=logits_trace, updates=updates)

    def step(self, state: Tensor, input_ids: Tensor, step_index: int, total_steps: int) -> StepOutput:
        raise NotImplementedError

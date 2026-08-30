"""Core Tiny Recursive Model compatible with the HRM-Mini training loop.

This keeps the recursive state update from the official TRM implementation,
while intentionally omitting puzzle-id embeddings and ACT/Q-learning halting.
"""

from typing import Any

import torch
from torch import Tensor, nn

from arch.layers import (
    Carry,
    CastedLinear,
    CastedScaledEmbedding,
    Transformer,
    TransformerConfig,
    trunc_normal_init_,
)


class TRMConfig(TransformerConfig):
    vocab_size: int

    H_cycles: int
    L_cycles: int
    bptt: bool

    forward_dtype: str


class TRM(nn.Module):
    """A single shared reasoning module that alternates L and H updates."""

    def __init__(self, config_dict: dict[str, Any]) -> None:
        super().__init__()
        config = TRMConfig(**config_dict)
        dtype = getattr(torch, config.forward_dtype)

        self.H_cycles = config.H_cycles
        self.L_cycles = config.L_cycles
        self.bptt = config.bptt

        # Unlike HRM, both state updates use exactly this one module.
        self.core = Transformer(config)
        self.embed = CastedScaledEmbedding(config.vocab_size, config.hidden_size, cast_to=dtype)
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)

        self.zH_init = nn.Buffer(
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=dtype)), persistent=True
        )
        self.zL_init = nn.Buffer(
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=dtype)), persistent=True
        )

    def forward(self, carry: Carry, input_ids: Tensor) -> tuple[Carry, Tensor]:
        x = self.embed(input_ids)
        z_H, z_L = carry["z_H"], carry["z_L"]

        if torch.is_grad_enabled() and self.bptt:
            # Full BPTT through all recursive updates, matching the project's
            # existing HRM/RT meaning of ``bptt=True``.
            for _ in range(self.H_cycles):
                for _ in range(self.L_cycles):
                    z_L = self.core(z_L + z_H + x)
                z_H = self.core(z_H + z_L)
        else:
            # Truncate all but the final L and H updates. This mirrors the
            # one-step-gradient path used by the existing recurrent models.
            with torch.no_grad():
                for _ in range(self.H_cycles - 1):
                    for _ in range(self.L_cycles):
                        z_L = self.core(z_L + z_H + x)
                    z_H = self.core(z_H + z_L)
                for _ in range(self.L_cycles - 1):
                    z_L = self.core(z_L + z_H + x)

            z_L = self.core(z_L + z_H + x)
            z_H = self.core(z_H + z_L)

        return dict(z_H=z_H.detach(), z_L=z_L.detach()), self.lm_head(z_H)

    @property
    def initial_carry(self) -> Carry:
        return dict(z_H=self.zH_init, z_L=self.zL_init)

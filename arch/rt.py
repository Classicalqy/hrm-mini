from collections.abc import Callable, Collection
from typing import Any, Literal

import torch
from torch import nn
from torch import Tensor

from arch.layers import CastedScaledEmbedding, CastedLinear, TransformerConfig, Transformer, Carry, trunc_normal_init_

class RecurrentTransformerConfig(TransformerConfig):
    vocab_size: int

    cycles: int
    bptt: bool

    forward_dtype: str

class RecurrentTransformer(nn.Module):
    def __init__(self, config_dict: dict[str, Any]) -> None:
        super().__init__()
        config = RecurrentTransformerConfig(**config_dict)
        dtype = getattr(torch, config.forward_dtype)

        self.cycles = config.cycles
        self.bptt = config.bptt

        # Backbone Layers
        self.core = Transformer(config)
        # I/O Layers
        self.embed = CastedScaledEmbedding(config.vocab_size, config.hidden_size, cast_to=dtype)
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)

        # Initial z
        self.z_init = nn.Buffer(trunc_normal_init_(torch.empty(config.hidden_size, dtype=dtype)), persistent=True)

    def forward(self, carry: Carry, input_ids: Tensor) -> tuple[Carry, Tensor]:
        x = self.embed(input_ids)

        # Forward iterations
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.bptt):
            z = carry["z"]
            for _i in range(self.cycles - 1):
                z = self.core(z + x)

        # 1-step grad
        z = self.core(z + x)
        return dict(z=z.detach()), self.lm_head(z)  # Ensure no gradient moves across carry

    def forward_with_trace(
        self,
        carry: Carry,
        input_ids: Tensor,
        trace_callback: Callable[[Literal["z"], Tensor], None],
        events: Collection[Literal["z"]] = ("z",),
    ) -> tuple[Carry, Tensor]:
        """Inference-equivalent forward pass that reports requested recurrent states.

        This mirrors HRM's trace API for deterministic rollout analysis while
        keeping the ordinary ``forward`` path unchanged.
        """
        invalid_events = set(events) - {"z"}
        if invalid_events:
            raise ValueError(f"Unsupported trace events: {sorted(invalid_events)}")
        x = self.embed(input_ids)
        z = carry["z"]
        for _ in range(self.cycles):
            z = self.core(z + x)
            if "z" in events:
                trace_callback("z", z)
        return dict(z=z.detach()), self.lm_head(z)

    @property
    def initial_carry(self) -> Carry:
        return dict(z=self.z_init)

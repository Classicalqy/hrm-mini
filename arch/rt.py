from typing import Any

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

    @property
    def initial_carry(self) -> Carry:
        return dict(z=self.z_init)

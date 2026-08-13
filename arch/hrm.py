from typing import Any, Literal

import torch
from torch import nn
from torch import Tensor

from arch.layers import CastedScaledEmbedding, CastedLinear, TransformerConfig, Transformer, Carry, trunc_normal_init_

class HRMConfig(TransformerConfig):
    vocab_size: int

    H_cycles: int
    L_cycles: int
    bptt: bool
    readout: Literal["h", "l", "hl"] = "h"

    forward_dtype: str

class HRM(nn.Module):
    def __init__(self, config_dict: dict[str, Any]) -> None:
        super().__init__()
        config = HRMConfig(**config_dict)
        dtype = getattr(torch, config.forward_dtype)

        self.H_cycles = config.H_cycles
        self.L_cycles = config.L_cycles
        self.bptt = config.bptt
        self.readout = config.readout

        # Backbone Layers
        self.H_level = Transformer(config)
        self.L_level = Transformer(config)
        # I/O Layers
        self.embed = CastedScaledEmbedding(config.vocab_size, config.hidden_size, cast_to=dtype)
        readout_size = config.hidden_size * (2 if self.readout == "hl" else 1)
        self.lm_head = CastedLinear(readout_size, config.vocab_size, bias=False)

        # Initial z
        self.zH_init = nn.Buffer(trunc_normal_init_(torch.empty(config.hidden_size, dtype=dtype)), persistent=True)
        self.zL_init = nn.Buffer(trunc_normal_init_(torch.empty(config.hidden_size, dtype=dtype)), persistent=True)

    def forward(self, carry: Carry, input_ids: Tensor) -> tuple[Carry, Tensor]:
        x = self.embed(input_ids)

        # Forward iterations
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.bptt):
            z_H, z_L = carry["z_H"], carry["z_L"]
            for _i in range(self.H_cycles * self.L_cycles - 1):
                z_L = self.L_level(z_L + z_H + x)
                if (_i + 1) % self.L_cycles == 0:
                    z_H = self.H_level(z_H + z_L)

        # 1-step grad
        z_L = self.L_level(z_L + z_H + x)
        z_H = self.H_level(z_H + z_L)
        if self.readout == "h":
            readout_state = z_H
        elif self.readout == "l":
            readout_state = z_L
        else:  # self.readout == "hl"; validated by HRMConfig.
            readout_state = torch.cat((z_H, z_L), dim=-1)

        return dict(z_H=z_H.detach(), z_L=z_L.detach()), self.lm_head(readout_state)  # Ensure no gradient moves across carry

    @property
    def initial_carry(self) -> Carry:
        return dict(z_H=self.zH_init, z_L=self.zL_init)

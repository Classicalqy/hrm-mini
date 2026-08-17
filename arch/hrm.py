from collections.abc import Callable
from typing import Any, Literal

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

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

    def readout_logits(self, z_H: Tensor, z_L: Tensor) -> Tensor:
        """Decode a pair of H/L states without changing the recurrent dynamics."""
        # At the first L update of a rollout z_H can still be the shared
        # [hidden] initial buffer, while z_L is already [batch, seq, hidden].
        # Broadcast it for intermediate trace readouts; ordinary final readouts
        # already have matching batch/sequence dimensions.
        if z_H.ndim < z_L.ndim:
            z_H = z_H.reshape((1,) * (z_L.ndim - z_H.ndim) + z_H.shape).expand_as(z_L)
        if self.readout == "h":
            readout_state = z_H
        elif self.readout == "l":
            readout_state = z_L
        else:  # self.readout == "hl"; validated by HRMConfig.
            readout_state = torch.cat((z_H, z_L), dim=-1)
        return self.lm_head(readout_state)

    def split_hl_readout_logits(self, z_H: Tensor, z_L: Tensor) -> tuple[Tensor, Tensor]:
        """Return additive H and L logit terms for an HL-readout model."""
        if self.readout != "hl":
            raise ValueError("split_hl_readout_logits is only defined for readout='hl'.")
        if z_H.ndim < z_L.ndim:
            z_H = z_H.reshape((1,) * (z_L.ndim - z_H.ndim) + z_H.shape).expand_as(z_L)
        hidden_size = z_H.shape[-1]
        weight = self.lm_head.weight.to(z_H.dtype)
        return F.linear(z_H, weight[:, :hidden_size]), F.linear(z_L, weight[:, hidden_size:])

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
        return dict(z_H=z_H.detach(), z_L=z_L.detach()), self.readout_logits(z_H, z_L)  # Ensure no gradient moves across carry

    def forward_with_trace(
        self,
        carry: Carry,
        input_ids: Tensor,
        trace_callback: Callable[[Literal["l", "h"], Tensor, Tensor], None],
    ) -> tuple[Carry, Tensor]:
        """Inference-equivalent forward pass that reports every L and H state update.

        The callback receives ``("l", z_H, z_L)`` after each L update and
        ``("h", z_H, z_L)`` after each H update. It is intended for no-grad
        trajectory analysis; the ordinary ``forward`` path remains unchanged.
        """
        x = self.embed(input_ids)
        z_H, z_L = carry["z_H"], carry["z_L"]
        for update_index in range(self.H_cycles * self.L_cycles):
            z_L = self.L_level(z_L + z_H + x)
            trace_callback("l", z_H, z_L)
            if (update_index + 1) % self.L_cycles == 0:
                z_H = self.H_level(z_H + z_L)
                trace_callback("h", z_H, z_L)

        return dict(z_H=z_H.detach(), z_L=z_L.detach()), self.readout_logits(z_H, z_L)

    @property
    def initial_carry(self) -> Carry:
        return dict(z_H=self.zH_init, z_L=self.zL_init)

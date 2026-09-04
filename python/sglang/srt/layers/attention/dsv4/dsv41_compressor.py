"""Low-ratio (C1/C2) compressor for DeepSeek V4.1.

sglang's `Compressor` (compressor.py) serves the V4 C4/C128 ratios with fused
`wkv_gate` / `ape` weights; V4.1 stores a plain `{norm, wkv, wgate}` compressor
on the kv_source layers only. This module owns no paged KV: the attention
integration decides where the pooled latent is cached.
"""

from __future__ import annotations

import torch
from torch import nn

from sglang.srt.layers.layernorm import RMSNorm


class DeepseekV41Compressor(nn.Module):
    def __init__(
        self, hidden_size: int, head_dim: int, compress_ratio: int, eps: float
    ):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.norm = RMSNorm(head_dim, eps=eps)
        # ratio 1 is a plain projection (checkpoint stores it in bf16); ratio > 1
        # softmax-pools in fp32, so those projections are fp32 in the checkpoint.
        proj_dtype = torch.float32 if compress_ratio > 1 else torch.bfloat16
        self.wkv = nn.Linear(hidden_size, head_dim, bias=False, dtype=proj_dtype)
        if compress_ratio > 1:
            self.wgate = nn.Linear(
                hidden_size, head_dim, bias=False, dtype=torch.float32
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [num_tokens, hidden] -> [num_tokens // ratio, head_dim], pre-RoPE;
        # whole groups only, decode-step accumulation belongs to the caller.
        ratio = self.compress_ratio
        if ratio == 1:  # one token per group: no pooling, no gate, no fp32
            return self.norm(self.wkv(x))

        x = x.float()
        kv = self.wkv(x)  # [n, head_dim]
        score = self.wgate(x)  # [n, head_dim]
        n = x.shape[0] - x.shape[0] % ratio
        kv = kv[:n].unflatten(0, (-1, ratio))  # [n//ratio, ratio, head_dim]
        score = score[:n].unflatten(0, (-1, ratio))
        kv = (kv * score.softmax(dim=1)).sum(dim=1)  # [n//ratio, head_dim]
        return self.norm(kv.to(x.dtype))

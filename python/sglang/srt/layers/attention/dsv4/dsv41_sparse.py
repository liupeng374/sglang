"""DeepSeek V4.1 low-ratio sparse attention (compress ratios 1 and 2), torch path.

Only kv_source layers compress; the layers that follow with the same ratio read
the source's latent pool. index_source layers score the shared latents and
publish top-k latent slots that the following layers reuse. Everything here is
plain torch for bring-up; the FlashMLA sparse kernels take over once the
compression metadata is ratio-generic.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import nn

from sglang.srt.distributed import tensor_model_parallel_all_reduce
from sglang.srt.layers.dsv41.norm import RMSNorm
from sglang.srt.layers.dsv41.quant import fake_quant_fp4
from sglang.srt.layers.linear import ColumnParallelLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix


class DSV41Runtime:
    """What the source layers hand to the layers after them within one forward.
    Layers run in order and every source writes before its consumers read."""

    def __init__(self):
        self.topk_slots: Optional[torch.Tensor] = None  # [T, topk] latent pool rows
        self.topk_valid: Optional[torch.Tensor] = None  # [T, topk] bool
        # Per-token request row and position of the current forward.
        self.req: Optional[torch.Tensor] = None
        self.pos: Optional[torch.Tensor] = None
        # One bool mask per request, in batch request order, over compressed positions.
        self.candidates: Optional[List[torch.Tensor]] = None


def token_req_indices(forward_batch) -> torch.Tensor:
    """req_pool_indices repeated once per token of the batch."""
    req = forward_batch.req_pool_indices.to(torch.int64)
    if forward_batch.forward_mode.is_decode():
        return req
    assert forward_batch.forward_mode.is_extend(), (
        "the V4.1 torch attention path serves extend and decode only"
    )
    return torch.repeat_interleave(req, forward_batch.extend_seq_lens.to(torch.int64))


def rope_tail(
    x: torch.Tensor, freqs: torch.Tensor, rope_dim: int, inverse: bool = False
) -> torch.Tensor:
    """Rotate the last rope_dim features of x [T, ..., D] with complex freqs [T, rope_dim // 2]."""
    head, tail = x[..., :-rope_dim], x[..., -rope_dim:]
    tc = torch.view_as_complex(tail.float().unflatten(-1, (-1, 2)).contiguous())
    f = freqs.conj() if inverse else freqs
    f = f.view(x.shape[0], *([1] * (x.ndim - 2)), rope_dim // 2)
    rotated = torch.view_as_real(tc * f).flatten(-2).to(x.dtype)
    return torch.cat([head, rotated], dim=-1)


def last_token_per_request(mask: torch.Tensor, req: torch.Tensor) -> torch.Tensor:
    """Among tokens selected by mask, keep only the last one of each request."""
    idx = mask.nonzero().squeeze(1)
    if idx.numel() == 0:
        return mask
    order = torch.argsort(idx)
    idx = idx[order]
    r = req[idx]
    is_last = torch.ones_like(idx, dtype=torch.bool)
    is_last[:-1] = r[:-1] != r[1:]
    out = torch.zeros_like(mask)
    out[idx[is_last]] = True
    return out


class DeepseekV41Compressor(nn.Module):
    """Pools compress_ratio consecutive tokens into one pre-RoPE KV latent.
    Ratio 1 is a plain bf16 projection; ratio 2 gates two tokens with a softmax
    over their fp32 scores."""

    def __init__(
        self, hidden_size: int, head_dim: int, compress_ratio: int, eps: float
    ):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.norm = RMSNorm(head_dim, eps)
        proj_dtype = torch.float32 if compress_ratio > 1 else torch.bfloat16
        self.wkv = nn.Linear(hidden_size, head_dim, bias=False, dtype=proj_dtype)
        if compress_ratio > 1:
            self.wgate = nn.Linear(
                hidden_size, head_dim, bias=False, dtype=torch.float32
            )

    def project(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.compress_ratio == 1:
            return self.wkv(x), None
        x = x.float()
        return self.wkv(x), self.wgate(x)

    def finish(self, kv: torch.Tensor) -> torch.Tensor:
        return self.norm(kv.to(torch.bfloat16))

    @staticmethod
    def pool_pairs(kv2: torch.Tensor, score2: torch.Tensor) -> torch.Tensor:
        """kv2, score2 [n, 2, D] fp32 -> [n, D]"""
        return (kv2 * score2.softmax(dim=1)).sum(dim=1)


class DeepseekV41Indexer(nn.Module):
    """Scores compressed positions with a small fp4 side attention. Only a
    kv_source layer owns index keys; the other index sources read the source's."""

    def __init__(
        self,
        config,
        layer_id: int,
        head_dim: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ):
        super().__init__()
        tp_size = get_parallel().tp_size
        self.n_heads = config.index_n_heads
        assert self.n_heads % tp_size == 0
        self.n_local_heads = self.n_heads // tp_size
        self.index_head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.owns_k = layer_id in config.kv_source_layers
        self.is_candidate_source = layer_id == config.candidate_source_layer
        self.uses_candidates = 0 <= config.candidate_source_layer < layer_id
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size
        self.softmax_scale = self.index_head_dim**-0.5
        self.wq_b = ColumnParallelLinear(
            config.q_lora_rank,
            self.n_heads * self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
        )
        self.weights_proj = ColumnParallelLinear(
            config.hidden_size,
            self.n_heads,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=add_prefix("weights_proj", prefix),
        )
        if self.owns_k:
            self.wk = nn.Linear(
                head_dim, self.index_head_dim, bias=False, dtype=torch.bfloat16
            )
            self.k_norm = RMSNorm(self.index_head_dim, config.rms_norm_eps)

    def index_keys(self, latent: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """Pre-RoPE latents [n, D] -> fp4-rounded index keys [n, index_head_dim]."""
        k = self.k_norm(self.wk(latent))
        return fake_quant_fp4(rope_tail(k, freqs, self.rope_head_dim))

    def queries(self, q_lora: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        q, _ = self.wq_b(q_lora)
        q = q.view(q.shape[0], self.n_local_heads, self.index_head_dim)
        return fake_quant_fp4(rope_tail(q, freqs, self.rope_head_dim))

    def head_weights(self, x: torch.Tensor) -> torch.Tensor:
        w, _ = self.weights_proj(x)
        return w * (self.softmax_scale * self.n_heads**-0.5)

    def scores(
        self, q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """q [t, H, d], k [n, d], weights [t, H] -> [t, n], reduced over all TP heads.
        Kept in bf16 up to the reduction, as the reference does."""
        s = torch.einsum("bhd,nd->bhn", q, k)
        s = (s.relu() * weights.unsqueeze(-1)).sum(dim=1)
        if get_parallel().tp_size > 1:
            s = tensor_model_parallel_all_reduce(s)
        return s.float()


def sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    valid: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    chunk: int = 512,
) -> torch.Tensor:
    """q [T, H, D], k [T, N, D] (one latent is both key and value), valid [T, N],
    attn_sink [H] fp32 -> [T, H, D]. Probabilities round to bf16 before the value
    sum, matching the reference kernel."""
    T, H, _ = q.shape
    outs = []
    sink = attn_sink.float().view(1, H, 1)
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        qs, ks = q[s:e].float(), k[s:e].float()
        scores = torch.einsum("bhd,bnd->bhn", qs, ks) * softmax_scale
        scores = scores.masked_fill(~valid[s:e, None, :], -torch.inf)
        # A finite floor keeps a row with no valid slot at an all-zero output.
        row_max = scores.amax(dim=-1, keepdim=True).clamp_min(-1e30)
        probs = torch.exp(scores - row_max)
        denom = probs.sum(dim=-1, keepdim=True) + torch.exp(sink - row_max)
        out = torch.einsum("bhn,bnd->bhd", probs.to(q.dtype).float(), ks) / denom
        outs.append(out.to(q.dtype))
    return torch.cat(outs, dim=0)

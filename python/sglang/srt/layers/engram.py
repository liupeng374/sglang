"""Engram: a gated n-gram hash memory written into the hc residual stream at a few layers.

Predecessor tokens come from the per-request n-gram token table the scheduler maintains
(forward_batch.ngram_embedding_info). The token map, hash multipliers and prime layout are
shared with the pure-torch oracle sglang.srt.layers.dsv41.engram, so the ids match bit for bit.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn

from sglang.srt.distributed import tensor_model_parallel_all_reduce
from sglang.srt.layers.dsv41.engram import (
    EngramLayout,
    build_compressed_token_map,
    compute_hash_multipliers,
)
from sglang.srt.layers.dsv41.quant import FP8_BLOCK_SIZE
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.schedule_batch import MM_PAD_SHIFT_VALUE
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_model, get_parallel, get_serving
from sglang.srt.utils import add_prefix
from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer


def build_engram_layout(config) -> Optional[EngramLayout]:
    layer_ids = tuple(config.engram_layer_ids)
    if not layer_ids:
        return None
    return EngramLayout.build(
        layer_ids=layer_ids,
        num_embeddings=tuple(config.engram_num_embeddings),
        max_ngram_size=config.engram_max_ngram_size,
        n_heads=config.engram_n_heads,
        head_dim=config.engram_head_dim,
        vocab_size=config.engram_vocab_size,
    )


def compute_engram_hash_ids(
    tokens: torch.Tensor,
    blocked: torch.Tensor,
    pad_id: int,
    token_map: torch.Tensor,
    multipliers: torch.Tensor,
    primes: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """tokens [T, n]: column 0 is the token itself, column s its s-th predecessor;
    blocked [T, n] marks look-back that ran off the sequence start. Returns
    [T, n_engram_layers, (n - 1) * n_heads] row ids into each layer's table."""
    compressed = torch.where(blocked, pad_id, token_map[tokens])
    products = compressed.unsqueeze(1) * multipliers
    # XOR the multiplied ids one look-back at a time: after step i the running value
    # is the (i + 1)-gram hash, bucketed by that n-gram size's primes.
    rolling, hashes = products[..., 0], []
    for i in range(1, tokens.shape[-1]):
        rolling = torch.bitwise_xor(rolling, products[..., i])
        hashes.append(rolling.unsqueeze(-1) % primes[:, i - 1])
    return torch.cat(hashes, dim=-1) + offsets


class EngramHasher(nn.Module):
    """Hash ids for every token of a forward batch, [T, n_engram_layers, n_hash_cols]."""

    def __init__(
        self,
        layout: EngramLayout,
        tokenizer,
        pad_id: int,
        compressed_vocab_size: int,
    ):
        super().__init__()
        self.max_ngram_size = layout.max_ngram_size
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        assert vocab_size == compressed_vocab_size, (
            f"the tokenizer normalizes to {vocab_size} distinct tokens but the config "
            f"expects {compressed_vocab_size}; every hash multiplier depends on it"
        )
        self.pad_id = token_map[pad_id]
        flat = [
            [p for per_ngram in layer for p in per_ngram] for layer in layout.primes
        ]
        offsets = np.array([np.cumsum([0, *sizes[:-1]]) for sizes in flat])
        multipliers = compute_hash_multipliers(
            layout.layer_ids, layout.max_ngram_size, vocab_size
        )
        self.register_buffer("token_map", torch.tensor(token_map), persistent=False)
        self.register_buffer("multipliers", multipliers, persistent=False)
        self.register_buffer("primes", torch.tensor(layout.primes), persistent=False)
        self.register_buffer("offsets", torch.tensor(offsets), persistent=False)

    @classmethod
    def from_config(cls, config, layout: EngramLayout) -> EngramHasher:
        # The compressed token map is built with the HF normalizers, so the HF
        # tokenizer backend is used here whatever the serving backend is.
        tokenizer = get_tokenizer(
            get_serving().tokenizer_path,
            tokenizer_mode=get_serving().tokenizer_mode,
            trust_remote_code=get_model().trust_remote_code,
            revision=get_model().revision,
            tokenizer_backend="huggingface",
        )
        result = cls(
            layout,
            tokenizer,
            config.engram_pad_id,
            config.engram_compressed_vocab_size,
        )
        result.image_token_id = (
            config.image_token_id
            if config.model_type == "deepseek_v4.1" and config.vision_n_layers > 0
            else None
        )
        return result

    def forward(
        self, input_ids: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        info = forward_batch.ngram_embedding_info
        assert info is not None, "engram reads predecessors from the n-gram token table"
        req = forward_batch.req_pool_indices.to(torch.int64)
        if not forward_batch.forward_mode.is_decode():
            req = torch.repeat_interleave(
                req, forward_batch.extend_seq_lens.to(torch.int64)
            )
        positions = forward_batch.positions.to(torch.int64)
        shifts = torch.arange(self.max_ngram_size, device=positions.device)
        lookback = positions.unsqueeze(-1) - shifts
        tokens = info.token_table[req.unsqueeze(-1), lookback.clamp_min(0)].to(
            torch.int64
        )
        tokens[:, 0] = input_ids
        blocked = lookback < 0
        if self.image_token_id is not None:
            # Previous chunks remain hashed in the scheduler's token table.
            tokens = tokens.masked_fill(
                tokens >= MM_PAD_SHIFT_VALUE, self.image_token_id
            )
            # Once a lookback hits an image, every older predecessor is PAD.
            blocked = (
                (blocked | (tokens == self.image_token_id))
                .to(torch.int32)
                .cummax(-1)
                .values.bool()
            )
        return compute_engram_hash_ids(
            tokens,
            blocked,
            self.pad_id,
            self.token_map,
            self.multipliers,
            self.primes,
            self.offsets,
        )


class EngramEmbedding(nn.Module):
    """One layer's fp8 hash table, sharded over rows across the TP group. Rows are
    dequantized with their e8m0 block scales on lookup and reduced across ranks."""

    def __init__(self, num_embeddings: int, dim: int):
        super().__init__()
        self.tp_size = get_parallel().tp_size
        assert num_embeddings % self.tp_size == 0, (num_embeddings, self.tp_size)
        self.rows = num_embeddings // self.tp_size
        self.row_start = get_parallel().tp_rank * self.rows
        self.weight = nn.Parameter(
            torch.empty(self.rows, dim, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.scale = nn.Parameter(
            torch.empty(self.rows, dim // FP8_BLOCK_SIZE, dtype=torch.float8_e8m0fnu),
            requires_grad=False,
        )
        self.weight.weight_loader = self._load_rows
        self.scale.weight_loader = self._load_rows

    def _load_rows(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight[self.row_start : self.row_start + self.rows])

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        local = indices - self.row_start
        owned = (local >= 0) & (local < self.rows)
        local = local.masked_fill(~owned, 0)
        rows = self.weight[local].float().unflatten(-1, (-1, FP8_BLOCK_SIZE))
        values = (rows * self.scale[local].float().unsqueeze(-1)).flatten(-2)
        values = values.to(torch.bfloat16).masked_fill(~owned.unsqueeze(-1), 0)
        if self.tp_size > 1:
            values = tensor_model_parallel_all_reduce(values)
        return values


def engram_gate(
    x: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    clamp_value: float,
) -> torch.Tensor:
    """x [T, hc_mult, dim]; kv [T, (hc_mult + 1) * dim] holds one key per hc copy
    followed by the shared value. Adds the gated value to every copy."""
    hc_mult, dim = x.shape[-2:]
    key, value = kv.split([hc_mult * dim, dim], dim=-1)
    key = key.float().unflatten(-1, (hc_mult, dim))
    weight = q_weight.float() * k_weight.float()
    h = x.float()
    # Normalized per (token, hc copy) over dim, not jointly over the copies.
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(
        key.square().mean(-1) + eps
    )
    dot = (h * weight * key).sum(-1) * rstd * dim**-0.5
    # Signed square root before the sigmoid, matching the training kernel.
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot))
    return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


class Engram(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int,
        layout: EngramLayout,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_hash_index = layout.layer_ids.index(layer_id)
        self.eps = config.rms_norm_eps
        self.clamp_value = 1e-6
        dim, hc_mult = config.hidden_size, config.hc_mult
        self.embed = EngramEmbedding(
            layout.num_embeddings[self.layer_hash_index], layout.head_dim
        )
        n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        self.wkv = ReplicatedLinear(
            n_hash_cols * layout.head_dim,
            dim * (hc_mult + 1),
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wkv", prefix),
        )
        self.q_weight = nn.Parameter(torch.ones(hc_mult, dim), requires_grad=False)
        self.k_weight = nn.Parameter(torch.ones(hc_mult, dim), requires_grad=False)

    def forward(self, x: torch.Tensor, hash_ids: torch.Tensor) -> torch.Tensor:
        """x [T, hc_mult, dim]; hash_ids [T, n_hash_cols] for this layer."""
        kv, _ = self.wkv(self.embed(hash_ids).flatten(-2))
        return engram_gate(
            x, kv, self.q_weight, self.k_weight, self.eps, self.clamp_value
        )

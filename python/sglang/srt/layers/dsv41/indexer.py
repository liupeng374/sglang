import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.dsv41.args import DeepseekV41Args
from sglang.srt.layers.dsv41.linear import Linear
from sglang.srt.layers.dsv41.norm import RMSNorm
from sglang.srt.layers.dsv41.quant import fake_quant_fp4
from sglang.srt.layers.dsv41.rope import apply_rotary_emb_tail
from sglang.srt.layers.dsv41.shared import SharedAttentionRuntime


def select_candidate_blocks(
    logits: torch.Tensor,
    compress_lens: torch.Tensor | int,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Level one of the two-level top-k: a bool mask over positions keeping the
    topk_blocks best-scoring blocks per query. Unreachable positions are already -inf
    in logits, so an all -inf block means not reachable yet; the block holding the
    query's newest position is always kept."""
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)

    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(num_blocks, device=logits.device) == last, torch.inf
    )

    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > -torch.inf
    )
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


class Indexer(nn.Module):
    """Scores compressed positions with a small fp4 side attention and keeps the
    index_topk best per query. Only a kv_source layer owns index keys; every other
    indexer reads them from the shared runtime."""

    def __init__(self, args: DeepseekV41Args, layer_id: int):
        super().__init__()
        self.owns_k = layer_id in args.kv_source_layers
        self.compress_ratio = args.compress_ratios[layer_id]
        self.is_candidate_source = layer_id == args.candidate_source_layer
        self.uses_candidates = 0 <= args.candidate_source_layer < layer_id
        self.candidate_topk_blocks = args.candidate_topk_blocks
        self.candidate_block_size = args.candidate_block_size
        self.n_heads = args.index_n_heads
        self.index_head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.softmax_scale = self.index_head_dim**-0.5
        self.wq_b = Linear(
            args.q_lora_rank,
            self.n_heads * self.index_head_dim,
            dtype=torch.float8_e4m3fn,
        )
        self.weights_proj = Linear(args.dim, self.n_heads, dtype=torch.bfloat16)
        if self.owns_k:
            self.wk = Linear(args.head_dim, self.index_head_dim, dtype=torch.bfloat16)
            self.k_norm = RMSNorm(self.index_head_dim, args.norm_eps)
            self.register_buffer(
                "k_cache",
                torch.zeros(
                    args.max_batch_size,
                    args.max_seq_len // self.compress_ratio,
                    self.index_head_dim,
                    dtype=torch.bfloat16,
                ),
                persistent=False,
            )

    def _publish_k(self, latent, start_pos, seqlen, freqs_cis, shared, bsz):
        ratio, rd = self.compress_ratio, self.rope_head_dim
        # A latent stands for the first token of its group: group j sits at position j * ratio.
        if start_pos == 0:
            freqs = freqs_cis[: seqlen - seqlen % ratio : ratio]
        else:
            freqs = freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
        k = self.k_norm(self.wk(latent))
        k = fake_quant_fp4(apply_rotary_emb_tail(k, rd, freqs))
        start = start_pos // ratio
        self.k_cache[:bsz, start : start + k.size(1)] = k
        shared.index_k = self.k_cache

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        latent: torch.Tensor | None,
        start_pos: int,
        offset: int,
        freqs_cis: torch.Tensor,
        shared: SharedAttentionRuntime,
    ) -> torch.Tensor:
        """latent is this layer's pre-RoPE compressed latent, None when the layer does
        not compress or its current group is incomplete. Returns int32 [b, s, topk]
        positions shifted by offset, -1 where unreachable."""
        bsz, seqlen, _ = x.size()
        ratio, rd, end_pos = self.compress_ratio, self.rope_head_dim, start_pos + seqlen

        if self.owns_k and latent is not None:
            self._publish_k(latent, start_pos, seqlen, freqs_cis, shared, bsz)

        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.index_head_dim))
        q = fake_quant_fp4(apply_rotary_emb_tail(q, rd, freqs_cis[start_pos:end_pos]))

        index_k = shared.index_k[:bsz, : end_pos // ratio]
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        index_score = torch.einsum("bshd,btd->bsht", q, index_k)
        index_score = (index_score.relu() * weights.unsqueeze(-1)).sum(dim=2)

        # A compressed position becomes visible once the query has passed its last token.
        if start_pos == 0:
            compress_lens = (
                torch.arange(1, seqlen + 1, device=x.device) // ratio
            ).unsqueeze(-1)
            index_score = index_score.masked_fill(
                torch.arange(seqlen // ratio, device=x.device) >= compress_lens,
                -torch.inf,
            )
        else:
            compress_lens = end_pos // ratio

        if self.is_candidate_source:
            shared.candidates = select_candidate_blocks(
                index_score,
                compress_lens,
                self.candidate_topk_blocks,
                self.candidate_block_size,
            )
        elif self.uses_candidates:
            index_score = index_score.masked_fill(~shared.candidates, -torch.inf)

        topk = min(self.index_topk, end_pos // ratio)
        idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        return torch.where(idxs < compress_lens, idxs + offset, -1).int()

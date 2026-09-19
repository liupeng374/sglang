"""MLA owner-scatter helpers (SGLANG_MLA_OWNER_SCATTER, see
dp_attention.mla_owner_scatter_enabled).

Neutral home so both the model (kimi_k3) and the attention backends
(hybrid_linear_attn_backend) can share the owned-view construction without a
model import cycle. Ownership is ``req_pool_idx % attn_tp_size`` — stable
across steps, so each request's KV lives (and is read) on exactly one rank.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Tuple

import torch

from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def _owner_of(req_pool_indices: torch.Tensor, tp: int) -> torch.Tensor:
    return req_pool_indices.to(torch.long) % tp


def mla_scatter_applies(forward_batch: ForwardBatch) -> bool:
    """Whether this forward runs the owner-scattered MLA path. MUST be the
    single source of truth shared by the model dispatch (kimi_k3) and the
    backend graph-metadata dispatch (hybrid out_graph), so the owned sub-batch
    and the model-side row selection can never disagree.

    Owner-scatter covers decode and uniform-chain target_verify only:

    - Extend/prefill keeps the full-batch path: with the replicated
      (full-head) weights its module output is already complete on every
      rank — no assembly needed, radix/prefix KV stays fully replicated,
      and the duplicated full-head compute is an accepted trade-off.
    - Ragged verify layouts and tree links (retrieve_next_token) are
      per-request structures the owned sub-batch does not subset; they fall
      back to the full-batch path too (correct, just duplicated compute).
    """
    mode = forward_batch.forward_mode
    if mode.is_decode():
        return True
    if mode.is_target_verify():
        spec = forward_batch.spec_info
        return (
            spec is not None
            and getattr(spec, "ragged_verify_layout", None) is None
            and getattr(spec, "retrieve_next_token", None) is None
        )
    return False


def _owned_rows_for_mode(
    forward_batch: ForwardBatch, owned_req: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Row indices (in the local batch) of the owned requests, for the three
    row layouts: decode (1 row/request), target_verify (uniform draft-token
    blocks), extend (rows grouped by extend_seq_lens)."""
    mode = forward_batch.forward_mode
    if mode.is_decode_or_idle():
        return owned_req
    if mode.is_target_verify():
        dt = forward_batch.spec_info.draft_token_num
        return (
            owned_req.unsqueeze(1) * dt
            + torch.arange(dt, device=device, dtype=owned_req.dtype)
        ).reshape(-1)
    lens = forward_batch.extend_seq_lens.to(torch.long)
    starts = torch.cumsum(lens, 0) - lens
    owned_lens = lens[owned_req]
    owned_starts = starts[owned_req]
    total = int(owned_lens.sum().item())
    if total == 0:
        return owned_req[:0]
    group = torch.repeat_interleave(
        torch.arange(owned_req.numel(), device=device), owned_lens
    )
    group_start = torch.cumsum(owned_lens, 0) - owned_lens
    return (
        owned_starts[group]
        + torch.arange(total, device=device)
        - group_start[group]
    )


def _subset_fb(forward_batch: ForwardBatch, owned_req: torch.Tensor, rows: torch.Tensor):
    """copy.copy view with per-request arrays subset to the owned requests and
    per-token arrays subset to the owned rows."""
    sub = copy.copy(forward_batch)
    n_owned = owned_req.numel()
    sub.batch_size = n_owned
    sub._original_batch_size = n_owned
    sub.req_pool_indices = forward_batch.req_pool_indices[owned_req].contiguous()
    if forward_batch.seq_lens is not None:
        sub.seq_lens = forward_batch.seq_lens[owned_req].contiguous()
    if forward_batch.seq_lens_cpu is not None:
        if torch.is_tensor(forward_batch.seq_lens_cpu):
            sub.seq_lens_cpu = forward_batch.seq_lens_cpu[owned_req].contiguous()
        else:
            sub.seq_lens_cpu = [forward_batch.seq_lens_cpu[i] for i in owned_req.tolist()]
    if forward_batch.positions is not None:
        sub.positions = forward_batch.positions[rows]
    if forward_batch.out_cache_loc is not None:
        sub.out_cache_loc = forward_batch.out_cache_loc[rows]
    if forward_batch.input_ids is not None:
        sub.input_ids = forward_batch.input_ids[rows]
    mode = forward_batch.forward_mode
    if mode.is_extend() and not mode.is_target_verify():
        if forward_batch.extend_seq_lens is not None:
            sub.extend_seq_lens = forward_batch.extend_seq_lens[owned_req].contiguous()
            sub.extend_start_loc = (
                torch.cumsum(sub.extend_seq_lens, 0) - sub.extend_seq_lens
            )
        if forward_batch.extend_seq_lens_cpu is not None:
            sub.extend_seq_lens_cpu = [
                forward_batch.extend_seq_lens_cpu[i] for i in owned_req.tolist()
            ]
        if forward_batch.extend_prefix_lens is not None:
            sub.extend_prefix_lens = forward_batch.extend_prefix_lens[
                owned_req
            ].contiguous()
        if forward_batch.extend_prefix_lens_cpu is not None:
            sub.extend_prefix_lens_cpu = [
                forward_batch.extend_prefix_lens_cpu[i] for i in owned_req.tolist()
            ]
    # target_verify keeps the shared spec_info object: the chain (topk=1)
    # verify attention path only reads draft_token_num, which is uniform per
    # request, so it stays valid for the owned subset.
    return sub


def mla_owned_view(forward_batch: ForwardBatch):
    """Eager per-step owned view: (owned_req_indices, owned_row_indices,
    sub_forward_batch) with compact (unpadded) subsets. Cached on the forward
    batch, keyed on the req_pool_indices tensor identity (fresh per step)
    plus the token-row count."""
    total_rows = (
        forward_batch.input_ids.shape[0]
        if forward_batch.input_ids is not None
        else -1
    )
    cached = getattr(forward_batch, "_mla_scatter_view", None)
    if (
        cached is not None
        and cached[0] is forward_batch.req_pool_indices
        and cached[1] == total_rows
    ):
        return cached[2], cached[3], cached[4]

    parallel = get_parallel()
    owner = _owner_of(forward_batch.req_pool_indices, parallel.attn_tp_size)
    owned_req = torch.nonzero(owner == parallel.attn_tp_rank, as_tuple=True)[0]
    rows = _owned_rows_for_mode(forward_batch, owned_req, owned_req.device)
    sub = _subset_fb(forward_batch, owned_req, rows)

    forward_batch._mla_scatter_view = (
        forward_batch.req_pool_indices,
        total_rows,
        owned_req,
        rows,
        sub,
    )
    return owned_req, rows, sub


def mla_graph_owned_view(forward_batch: ForwardBatch, out_cache_loc_buf: torch.Tensor):
    """Graph capture/replay owned view: the owned subset FRONT-PACKED and
    zero-padded to the graph bucket (forward_batch.batch_size), so the
    captured full-attention kernels see a static-shaped batch where rows
    [0, n_rows) are this rank's owned requests and the tail is inert padding
    (zero seq_lens / out_cache_loc -> slot-0 writes, zero block tables).

    out_cache_loc_buf is the backend's persistent static buffer (captured by
    pointer inside the graph kernel launch); it is refreshed in place here.

    Returns (sub_fb, owned_rows, n_rows).
    """
    assert (
        forward_batch.forward_mode.is_decode_or_idle()
        or forward_batch.forward_mode.is_target_verify()
    ), "graph owner-scatter supports decode/target_verify only"
    parallel = get_parallel()
    bucket = forward_batch.batch_size
    device = forward_batch.req_pool_indices.device
    owner = _owner_of(forward_batch.req_pool_indices, parallel.attn_tp_size)
    owned_req = torch.nonzero(owner == parallel.attn_tp_rank, as_tuple=True)[0]
    owned_rows = _owned_rows_for_mode(forward_batch, owned_req, device)
    n_rows = owned_rows.numel()

    def pad_front(t: torch.Tensor, fill=0) -> torch.Tensor:
        out = t.new_full((bucket,) + tuple(t.shape[1:]), fill)
        out[:n_rows] = t[:n_rows]
        return out

    sub = copy.copy(forward_batch)
    sub.batch_size = bucket
    sub._original_batch_size = bucket
    sub.req_pool_indices = pad_front(forward_batch.req_pool_indices)
    if forward_batch.seq_lens is not None:
        sub.seq_lens = pad_front(forward_batch.seq_lens)
    if forward_batch.seq_lens_cpu is not None:
        if torch.is_tensor(forward_batch.seq_lens_cpu):
            sub.seq_lens_cpu = pad_front(forward_batch.seq_lens_cpu)
        else:
            sub.seq_lens_cpu = (
                forward_batch.seq_lens_cpu[:n_rows]
                + [0] * (bucket - n_rows)
            )
    if forward_batch.positions is not None:
        sub.positions = pad_front(forward_batch.positions)
    if forward_batch.input_ids is not None:
        sub.input_ids = pad_front(forward_batch.input_ids)
    # The KV-write targets are captured by pointer inside the graph: refresh
    # the backend's persistent buffer in place (owned rows front-packed,
    # tail -> reserved slot 0).
    if forward_batch.out_cache_loc is not None:
        out_cache_loc_buf[:n_rows].copy_(forward_batch.out_cache_loc[owned_rows])
        out_cache_loc_buf[n_rows:].zero_()
        sub.out_cache_loc = out_cache_loc_buf[:bucket]
    return sub, owned_rows, n_rows


def mla_scatter_row_partition(
    sub_out: torch.Tensor,
    rows: torch.Tensor,
    full_shape: Tuple[int, ...],
    group,
) -> torch.Tensor:
    """Eager assembly: place the owned rows' attention outputs at their
    original positions (zeros elsewhere) and sum the per-owner partition with
    an attn-tp all-reduce — each row is produced on exactly one rank, so the
    sum is exact. The collective shape (full local batch) is identical on
    every rank, which keeps eager and graph paths on the same contract."""
    out = sub_out.new_zeros(full_shape)
    out.index_add_(0, rows, sub_out)
    return group.all_reduce(out)

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


# ---------------------------------------------------------------------------
# Token-shard mode flag
#
# When the SP-MoE pipeline keeps its output sharded (sp_attn_res = True:
# k3_sp_collective + SGLANG_K3_SP_ATTN_RES), MLA consumes the 1/attn_tp token
# shard directly — no all-gather before attention, no reduce-scatter after.
# The model sets this once from its sp_attn_res computation; the backend's
# graph-metadata planner reads it to pick the shard-sized sub_fb (vs the
# full-bucket owner-scatter sub_fb used when sp_attn_res is False).
# ---------------------------------------------------------------------------
_TOKEN_SHARD_ACTIVE: bool = False


def set_mla_token_shard_active(value: bool) -> None:
    global _TOKEN_SHARD_ACTIVE
    _TOKEN_SHARD_ACTIVE = value


def mla_token_shard_active() -> bool:
    """Whether MLA consumes the SP-MoE token shard directly (no all-gather)."""
    return _TOKEN_SHARD_ACTIVE


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
    zero-padded to the graph bucket, so the captured full-attention kernels
    see a static-shaped batch where the front rows are this rank's owned
    requests and the tail is inert padding (zero seq_lens / out_cache_loc ->
    slot-0 writes, zero block tables).

    Two granularities (verify packs draft-token blocks per request):
    - request-level arrays (req_pool_indices / seq_lens / seq_lens_cpu) pad to
      forward_batch.batch_size (the graph's request bucket) with n_owned_req;
    - row-level arrays (positions / input_ids / out_cache_loc) pad to the
      model row bucket with n_rows (= owned requests x draft_token_num for
      verify). out_cache_loc_buf is the backend's persistent static buffer
      (captured by pointer inside the graph kernel launch); refreshed in
      place here.

    Returns (sub_fb, owned_rows, n_rows).
    """
    assert (
        forward_batch.forward_mode.is_decode_or_idle()
        or forward_batch.forward_mode.is_target_verify()
    ), "graph owner-scatter supports decode/target_verify only"
    parallel = get_parallel()
    req_bucket = forward_batch.batch_size
    row_bucket = (
        forward_batch.input_ids.shape[0]
        if forward_batch.input_ids is not None
        else req_bucket
    )
    device = forward_batch.req_pool_indices.device
    owner = _owner_of(forward_batch.req_pool_indices, parallel.attn_tp_size)
    owned_req = torch.nonzero(owner == parallel.attn_tp_rank, as_tuple=True)[0]
    owned_rows = _owned_rows_for_mode(forward_batch, owned_req, device)
    n_req = owned_req.numel()
    n_rows = owned_rows.numel()

    def pad_front_req(t: torch.Tensor) -> torch.Tensor:
        out = t.new_full((req_bucket,) + tuple(t.shape[1:]), 0)
        out[:n_req] = t[:n_req]
        return out

    def pad_front_row(t: torch.Tensor) -> torch.Tensor:
        out = t.new_full((row_bucket,) + tuple(t.shape[1:]), 0)
        out[:n_rows] = t[:n_rows]
        return out

    sub = copy.copy(forward_batch)
    sub.batch_size = req_bucket
    sub._original_batch_size = req_bucket
    sub.req_pool_indices = pad_front_req(forward_batch.req_pool_indices)
    if forward_batch.seq_lens is not None:
        sub.seq_lens = pad_front_req(forward_batch.seq_lens)
    if forward_batch.seq_lens_cpu is not None:
        if torch.is_tensor(forward_batch.seq_lens_cpu):
            sub.seq_lens_cpu = pad_front_req(forward_batch.seq_lens_cpu)
        else:
            sub.seq_lens_cpu = (
                list(forward_batch.seq_lens_cpu[:n_req]) + [0] * (req_bucket - n_req)
            )
    if forward_batch.positions is not None:
        sub.positions = pad_front_row(forward_batch.positions)
    if forward_batch.input_ids is not None:
        sub.input_ids = pad_front_row(forward_batch.input_ids)
    # The KV-write targets are captured by pointer inside the graph: refresh
    # the backend's persistent buffer in place (owned rows front-packed,
    # tail -> reserved slot 0).
    if forward_batch.out_cache_loc is not None:
        loc = forward_batch.out_cache_loc
        n_loc = loc.shape[0]
        if n_loc == 0:
            # IDLE / empty replay batch (e.g. an idle DP rank with 0 tokens):
            # no live KV write targets; slot 0 (inert) for all owned rows so
            # front-packing the owned subset stays in-bounds against the graph
            # bucket.
            out_cache_loc_buf[:n_rows].zero_()
        else:
            safe_rows = owned_rows.clamp(max=n_loc - 1)
            picked = loc[safe_rows].to(out_cache_loc_buf.dtype)
            out_of_range = owned_rows >= n_loc
            if bool(out_of_range.any()):
                picked = torch.where(
                    out_of_range, torch.zeros_like(picked), picked
                )
            out_cache_loc_buf[:n_rows].copy_(picked)
        out_cache_loc_buf[n_rows:].zero_()
        sub.out_cache_loc = out_cache_loc_buf[:row_bucket]
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


# ---------------------------------------------------------------------------
# Token-shard views (SP-MoE token shard, 1/attn_tp contiguous rows)
#
# When MLA receives the SP-MoE reduce-scatter output directly (no all-gather),
# the hidden_states is already 1/attn_tp of the full batch — a contiguous token
# range.  These helpers build the matching sub-forward-batch so the attention
# backend plans metadata for the shard's requests only.
# ---------------------------------------------------------------------------

def _shard_req_rows(forward_batch: ForwardBatch):
    """Contiguous (req_indices, row_indices) for this rank's token shard."""
    parallel = get_parallel()
    rank = parallel.attn_tp_rank
    tp = parallel.attn_tp_size
    mode = forward_batch.forward_mode
    device = forward_batch.req_pool_indices.device

    if mode.is_decode_or_idle():
        total = forward_batch.batch_size
        shard = total // tp
        lo = rank * shard
        req_indices = torch.arange(
            lo, lo + shard, device=device, dtype=torch.long
        )
        rows = req_indices  # decode: 1 row per request
    elif mode.is_target_verify():
        total_rows = (
            forward_batch.input_ids.shape[0]
            if forward_batch.input_ids is not None
            else 0
        )
        shard_rows = total_rows // tp
        lo = rank * shard_rows
        rows = torch.arange(
            lo, lo + shard_rows, device=device, dtype=torch.long
        )
        dt = forward_batch.spec_info.draft_token_num
        req_indices = torch.arange(
            lo // dt, lo // dt + shard_rows // dt,
            device=device, dtype=torch.long,
        )
    else:
        raise RuntimeError(
            "token-shard view supports decode/target_verify only"
        )
    return req_indices, rows


def mla_shard_view(forward_batch: ForwardBatch):
    """Eager per-step token-shard view: (req_indices, row_indices,
    sub_forward_batch) for the contiguous 1/attn_tp token range produced by
    the SP-MoE reduce-scatter.

    Unlike mla_owned_view (which selects by req_pool_idx % attn_tp), this
    selects the contiguous range [rank*shard, (rank+1)*shard). Used when MLA
    receives the SP-MoE token shard directly (no all-gather): each rank
    computes full-head MLA attention for its 1/attn_tp token subset, and the
    output is already correct (MLA replication → no reduce needed).
    """
    total_rows = (
        forward_batch.input_ids.shape[0]
        if forward_batch.input_ids is not None
        else -1
    )
    cached = getattr(forward_batch, "_mla_shard_view", None)
    if (
        cached is not None
        and cached[0] is forward_batch.req_pool_indices
        and cached[1] == total_rows
    ):
        return cached[2], cached[3], cached[4]

    req_indices, rows = _shard_req_rows(forward_batch)
    sub = _subset_fb(forward_batch, req_indices, rows)

    forward_batch._mla_shard_view = (
        forward_batch.req_pool_indices,
        total_rows,
        req_indices,
        rows,
        sub,
    )
    return req_indices, rows, sub


def mla_graph_shard_view(
    forward_batch: ForwardBatch, out_cache_loc_buf: torch.Tensor
):
    """Graph capture/replay token-shard view: a compact SHARD-SIZED sub
    forward batch matching the 1/attn_tp contiguous token range produced by
    the SP-MoE reduce-scatter.

    Unlike mla_graph_owned_view (which front-packs owned rows into the full
    graph bucket with zero padding), this prepares a sub_fb whose batch_size
    and row count are exactly the shard size, because under sp_attn_res the
    graph captures MLA attention kernels at shard size (the SP-MoE
    reduce-scatter output shape).

    out_cache_loc_buf is the backend's persistent KV-write-target buffer
    (allocated at full-bucket size); only the first n_rows elements are
    used as a view.

    Returns (sub_fb, shard_row_indices, n_rows).
    """
    assert (
        forward_batch.forward_mode.is_decode_or_idle()
        or forward_batch.forward_mode.is_target_verify()
    ), "graph token-shard view supports decode/target_verify only"
    parallel = get_parallel()
    rank = parallel.attn_tp_rank
    tp = parallel.attn_tp_size
    req_bucket = forward_batch.batch_size
    row_bucket = (
        forward_batch.input_ids.shape[0]
        if forward_batch.input_ids is not None
        else req_bucket
    )
    device = forward_batch.req_pool_indices.device
    mode = forward_batch.forward_mode

    if mode.is_decode_or_idle():
        shard_req = req_bucket // tp
        shard_row = row_bucket // tp
    else:  # target_verify
        shard_row = row_bucket // tp
        dt = forward_batch.spec_info.draft_token_num
        shard_req = shard_row // dt

    lo_req = rank * shard_req
    lo_row = rank * shard_row
    n_req = shard_req
    n_rows = shard_row
    shard_rows = torch.arange(
        lo_row, lo_row + n_rows, device=device, dtype=torch.long
    )

    sub = copy.copy(forward_batch)
    sub.batch_size = n_req
    sub._original_batch_size = n_req
    sub.req_pool_indices = forward_batch.req_pool_indices[
        lo_req : lo_req + n_req
    ].contiguous()
    if forward_batch.seq_lens is not None:
        sub.seq_lens = forward_batch.seq_lens[
            lo_req : lo_req + n_req
        ].contiguous()
    if forward_batch.seq_lens_cpu is not None:
        if torch.is_tensor(forward_batch.seq_lens_cpu):
            sub.seq_lens_cpu = forward_batch.seq_lens_cpu[
                lo_req : lo_req + n_req
            ].contiguous()
        else:
            sub.seq_lens_cpu = list(
                forward_batch.seq_lens_cpu[lo_req : lo_req + n_req]
            )
    if forward_batch.positions is not None:
        sub.positions = forward_batch.positions[
            lo_row : lo_row + n_rows
        ].contiguous()
    if forward_batch.input_ids is not None:
        sub.input_ids = forward_batch.input_ids[
            lo_row : lo_row + n_rows
        ].contiguous()
    if forward_batch.out_cache_loc is not None:
        loc = forward_batch.out_cache_loc
        n_loc = loc.shape[0]
        if n_loc == 0:
            out_cache_loc_buf[:n_rows].zero_()
        else:
            safe_end = min(lo_row + n_rows, n_loc)
            actual_n = max(0, safe_end - lo_row)
            if actual_n > 0:
                out_cache_loc_buf[:actual_n].copy_(
                    loc[lo_row:safe_end].to(out_cache_loc_buf.dtype)
                )
            if actual_n < n_rows:
                out_cache_loc_buf[actual_n:n_rows].zero_()
        sub.out_cache_loc = out_cache_loc_buf[:n_rows]
    return sub, shard_rows, n_rows

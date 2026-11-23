"""Rolling packed-C8 KV prefetch for HiSparse IndexShare layers."""

from __future__ import annotations

import torch
import torch_npu
from sgl_kernel_npu.sparsity_driven_kv_offload import unidex_copy_inplace


def selection_masks(topk, positions, seq_lens, requests, request_rows):
    """Split causal top-k into host history and this invocation's KV rows."""
    batch, width = positions.shape
    sources = topk.reshape(batch, width, -1)
    live = (
        (seq_lens[:batch] > 0)
        & (requests[:batch] > 0)
        & (requests[:batch] < request_rows)
    )
    valid = (
        live[:, None, None]
        & (sources >= 0)
        & (sources <= positions[:, :, None])
    )
    historical = valid & (sources < positions[:, :1, None])
    current = valid & ~historical
    return historical.reshape_as(topk), current.reshape_as(topk)


class RollingC8SharedPrefetch:
    """Prefetch one shared layer at a time into a single reusable workspace."""

    def __init__(self, manager):
        self.manager = manager
        self.groups = {}
        self.layer_to_anchor = {}
        anchor = None
        for layer, shared in enumerate(manager.budget.shared_index_layers):
            if shared:
                if anchor is not None:
                    self.groups.setdefault(anchor, []).append(layer)
                    self.layer_to_anchor[layer] = anchor
            else:
                anchor = layer

        self.stream = torch.npu.Stream()
        self.buffer = (
            manager.budget.prefetch_spec.allocate(manager.device)
            if manager.budget.prefetch_spec.count
            else None
        )
        self.ready = torch.npu.Event()
        self.pending = None

    def start_anchor(self, anchor, batch, plan):
        group = self.groups.get(anchor, ())
        if group:
            if self.pending is not None:
                # Recover from a forward that exited before consuming its
                # shared layer. Do not let that plan leak into a new batch.
                self.stream.synchronize()
                self.pending = None
            positions = batch.positions[: plan.source_ids.shape[0]].reshape(
                batch.batch_size, plan.query_width
            )
            self._prefetch(group[0], batch, plan.source_ids, positions)

    def _prefetch(self, layer, batch, topk, positions):
        if self.pending is not None:
            raise RuntimeError("Rolling C8 prefetch buffer is still in use.")
        manager = self.manager
        rows, count = topk.shape
        width = positions.shape[1]
        if self.buffer is None:
            raise RuntimeError("C8 shared layers have no reserved prefetch workspace.")
        if rows != batch.batch_size * width or rows * count > self.buffer.shape[0]:
            raise RuntimeError("C8 shared prefetch exceeds its planned workspace.")

        producer = torch.npu.current_stream(manager.device)
        self.stream.wait_stream(producer)
        for tensor in (topk, positions, batch.seq_lens, batch.req_pool_indices):
            tensor.record_stream(self.stream)
        with torch.npu.stream(self.stream):
            historical, _ = selection_masks(
                topk, positions, batch.seq_lens, batch.req_pool_indices, manager.size
            )
            requests = (
                batch.req_pool_indices[:, None, None]
                .expand(-1, width, count)
                .reshape(rows, count)
            )
            sources = manager.host_source_indices(requests, topk).reshape(-1)
            destinations = torch.arange(
                rows * count, dtype=torch.long, device=topk.device
            )
            selected = self.buffer[: rows * count]
            selected.zero_()
            unidex_copy_inplace(
                manager.host_kv_buffer[layer],
                selected,
                sources,
                destinations,
                historical.reshape(-1),
                2,
                2,
                block_dim=24,
                src_ptr=manager.dev_ptr_list[layer],
            )
            self.ready.record(self.stream)
        self.pending = (layer, batch, rows, width, count)

    def attention(self, layer, q, q_rope, layer_obj, batch, plan, current_inputs):
        local = layer_obj.layer_id - self.manager.start_layer
        pending = self.pending
        if (
            pending is None
            or pending[0] != local
            or pending[1] is not batch
            or pending[2] != plan.source_ids.shape[0]
            or pending[3] != plan.query_width
            or pending[4] != plan.source_ids.shape[1]
        ):
            self.pending = None
            raise RuntimeError(
                "C8 shared layer has no matching anchor/rolling prefetch."
            )
        self.pending = None
        _, _, rows, width, count = pending
        batch_size = batch.batch_size
        stream = torch.npu.current_stream(self.manager.device)
        stream.wait_event(self.ready)

        selected = self.buffer[: rows * count]
        selected.record_stream(stream)
        historical, current_mask = selection_masks(
            plan.source_ids,
            batch.positions[:rows].reshape(batch_size, width),
            batch.seq_lens,
            batch.req_pool_indices,
            self.manager.size,
        )
        current, _, current_valid = current_inputs
        if current is not None:
            topk = plan.source_ids.reshape(batch_size, width, count)
            row_positions = batch.positions[:rows].reshape(batch_size, width)
            offsets = topk - row_positions[:, :1, None]
            current_rows = (
                torch.arange(batch_size, device=q.device)[:, None, None] * width
                + offsets
            ).reshape(-1).long()
            destinations = torch.arange(
                rows * count, device=q.device, dtype=torch.long
            )
            current_mask &= current_valid[:rows].reshape(batch_size, width, 1).expand(
                -1, -1, count
            ).reshape(rows, count)
            unidex_copy_inplace(
                current,
                selected,
                current_rows,
                destinations,
                current_mask.reshape(-1),
                1,
                2,
                block_dim=24,
            )

        valid = historical | current_mask
        sparse_indices = torch.where(
            valid,
            torch.arange(count, dtype=torch.int32, device=q.device)[None, :],
            -1,
        )
        # Empty/padded rows use a harmless physical slot and are zeroed below.
        sparse_indices[:, 0] = torch.where(
            valid.any(dim=1),
            sparse_indices[:, 0],
            torch.zeros_like(sparse_indices[:, 0]),
        )
        page_size = self.manager.page_size
        if count % page_size:
            raise RuntimeError(
                "C8 shared TopK must be aligned to the physical page size."
            )
        key = selected.view(-1, page_size, 1, 656).view(torch.float8_e4m3fn)
        block_table = torch.arange(
            rows * (count // page_size), dtype=torch.int32, device=q.device
        ).reshape(rows, -1)
        query_ends = torch.arange(1, rows + 1, dtype=torch.int32, device=q.device)
        kv_lengths = torch.full((rows,), count, dtype=torch.int32, device=q.device)

        heads = q.shape[1]
        query = torch.cat((q[:rows], q_rope[:rows]), dim=-1)
        padded_heads = self.manager.budget.copy_sfa_query_heads
        if heads < padded_heads:
            query = torch.nn.functional.pad(query, (0, 0, 0, padded_heads - heads))
        output = torch_npu.npu_kv_quant_sparse_flash_attention(
            query=query.contiguous(),
            key=key,
            value=key,
            sparse_indices=sparse_indices.reshape(rows, 1, count),
            scale_value=layer_obj.scaling,
            key_quant_mode=2,
            value_quant_mode=2,
            key_dequant_scale=None,
            value_dequant_scale=None,
            actual_seq_lengths_query=query_ends,
            actual_seq_lengths_kv=kv_lengths,
            block_table=block_table,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
            attention_mode=2,
            quant_scale_repo_mode=1,
            tile_size=page_size,
            rope_head_dim=q_rope.shape[-1],
        )
        output = output[0] if isinstance(output, tuple) else output
        live = plan.valid_rows[:, None].expand(-1, width).reshape(rows)
        output = output[:, :heads].reshape(rows, -1)
        output.masked_fill_(~live[:, None], 0)

        anchor = self.layer_to_anchor[local]
        group = self.groups[anchor]
        next_index = group.index(local) + 1
        if next_index < len(group):
            # The next copy may overwrite the single workspace only after this
            # layer's SFA has finished reading it. It then overlaps later-layer
            # projection/MLP work on the consumer stream.
            self.stream.wait_stream(stream)
            next_layer = group[next_index]
            positions = batch.positions[:rows].reshape(batch_size, width)
            self._prefetch(next_layer, batch, plan.source_ids, positions)
        return torch.nn.functional.pad(output, (0, 0, 0, q.shape[0] - rows))

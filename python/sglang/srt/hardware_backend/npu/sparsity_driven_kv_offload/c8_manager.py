"""Ascend950 payload and FP8 indexer adapters for the shared offload manager."""

import torch
from sgl_kernel_npu.sparsity_driven_kv_offload import unidex_copy_inplace

from sglang.kernels.ops.kvcache.npu_sparse_kv import prepare_sparse_kv_rows
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_budget import (
    C8_ROW_BYTES,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_lim import (
    LIMDecodeAdapter,
    LIMSelection,
    sparse_query_width,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_manager import (
    FusedSparseKVCacheManager,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.memfabric_pool import (
    MemFabricKVPool,
)
from sglang.srt.runtime_context import get_parallel


def _copy_packed(source, destination, src, dst, valid):
    unidex_copy_inplace(
        source.view(-1, C8_ROW_BYTES),
        destination.view(-1, C8_ROW_BYTES),
        src.reshape(-1).long(),
        dst.reshape(-1).long(),
        valid.reshape(-1),
        1,
        1,
        block_dim=48,
        src_ptr=source.data_ptr() if source.is_cpu else None,
        dst_ptr=destination.data_ptr() if destination.is_cpu else None,
    )


class PackedMemFabricKVPool(MemFabricKVPool):
    def copy_rows(self, sources, destinations, source_rows, destination_rows, valid):
        (source,), (destination,) = sources, destinations
        if source_rows.numel():
            _copy_packed(source, destination, source_rows, destination_rows, valid)


class C8DecodeAdapter(LIMDecodeAdapter):
    @staticmethod
    def _load_operator():
        return torch.ops.npu.npu_quant_lightning_indexer

    def select(
        self,
        manager,
        *,
        query,
        weights,
        key,
        block_table,
        forward_batch,
        layer_id,
        query_scale,
        key_scale,
    ):
        layer = layer_id - manager.start_layer
        batch, width = self.prepare(manager, block_table, forward_batch, layer)
        rows = batch * width
        lengths = self._pre_lengths[:batch]
        topk = self.operator(
            query=query[:rows].contiguous(),
            key=key,
            weights=weights[:rows].contiguous(),
            query_dequant_scale=query_scale[:rows].contiguous(),
            key_dequant_scale=key_scale,
            actual_seq_lengths_query=self.query_ends[width - 1, :batch],
            actual_seq_lengths_key=lengths,
            block_table=self.block_table[:batch],
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=manager.sparse_topk,
            sparse_mode=3,
            query_quant_mode=0,
            key_quant_mode=0,
        )
        source_ids = self.source_ids[:rows]
        source_ids.copy_(topk.reshape(rows, manager.sparse_topk))
        return self.finish_full_topk(layer, batch, width)

    def finish_full_topk(self, layer_index, batch, width):
        """Publish raw top-k and request validity without building cache slots."""
        rows = batch * width
        self.last_prefix[layer_index].index_copy_(
            0, self.resident_safe_slots[:batch].long(), self._pre_next_prefix[:batch]
        )
        return LIMSelection(
            source_ids=self.source_ids[:rows],
            device_slots=None,
            topk_misses=None,
            miss_sources=None,
            miss_slots=None,
            miss_counts=None,
            sparse_rows=self._pre_sparse[:batch],
            valid_rows=self._pre_valid[:batch],
            request_safe_slots=self.request_safe_slots[:batch],
            resident_safe_slots=self.resident_safe_slots[:batch],
            query_width=width,
            owner_layer=layer_index,
        )


class C8SparseKVCacheManager(FusedSparseKVCacheManager):
    packed_c8 = True

    @staticmethod
    def _load_copy_sfa_operator():
        # C8 attention now reads the dense anchor cache or rolling selected-KV
        # workspace directly; neither path invokes CopySFA.
        return None

    def _make_decode_adapter(self):
        return C8DecodeAdapter(self)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_kv_prefetch = None
        if any(self.budget.shared_index_layers):
            from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.c8_resident import (
                RollingC8SharedPrefetch,
            )

            self.shared_kv_prefetch = RollingC8SharedPrefetch(self)

    def start_shared_prefetch(self, anchor, forward_batch, plan):
        if self.shared_kv_prefetch is not None:
            self.shared_kv_prefetch.start_anchor(anchor, forward_batch, plan)

    def has_shared_consumers(self, anchor):
        return bool(
            self.shared_kv_prefetch is not None
            and self.shared_kv_prefetch.groups.get(anchor)
        )

    def _init_host_pool(self):
        self.host_pool = self.mf_host_pool = PackedMemFabricKVPool(
            tokens=self.host_token_capacity,
            layers=self.layer_num,
            page_size=self.page_size,
            head_num=1,
            widths=(C8_ROW_BYTES,),
            dtype=torch.int8,
            device=self.device,
            group=get_parallel().attn_tp_group,
        )
        (self.host_kv_buffer,) = self.host_pool.buffers
        self.dev_ptr_list = [x.data_ptr() for x in self.host_kv_buffer]

    def invalidate_packed_request(self, request_id):
        resident_id = (
            request_id
            if self.request_to_resident is None
            else self._resident_rows_by_request[request_id]
        )
        if resident_id:
            self.reset_requests([request_id], [resident_id])

    def offload_v2(self, k, k_rope, layer, forward_batch):
        layer_index = layer.layer_id - self.start_layer
        current = self.paged_kv_cache._pack_dsa_fp8_kv_cache(k, k_rope).view(torch.int8)
        prefill = forward_batch.forward_mode.is_extend_without_speculative()
        src, dst, positions, valid = prepare_sparse_kv_rows(
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            None if prefill else forward_batch.positions,
            forward_batch.out_cache_loc,
            tokens=current.shape[0],
            device=current.device,
            request_rows=self.size,
            context_len=self.logical_max_context_len,
            host_capacity=self.host_token_capacity,
            query_width=1 if prefill else sparse_query_width(forward_batch),
            extend_seq_lens=forward_batch.extend_seq_lens if prefill else None,
            extend_prefix_lens=forward_batch.extend_prefix_lens if prefill else None,
            compact_prefill=prefill
            and sum(forward_batch.extend_seq_lens_cpu[: forward_batch.batch_size])
            == current.shape[0],
        )
        if self.host_pool.rank == 0:
            _copy_packed(current, self.host_kv_buffer[layer_index], src, dst, valid)
        self.host_pool.publish()
        if prefill:
            requests = forward_batch.req_pool_indices.long()
            resident_ids = (
                requests
                if self.request_to_resident is None
                else self.request_to_resident.index_select(0, requests).long()
            )
            self.lim_adapter.last_prefix[layer_index].index_fill_(
                0, resident_ids, -1
            )
        return current, positions, valid

    def reset_requests(self, req_ids, resident_ids=None):
        if not req_ids:
            return
        resident_ids = req_ids if resident_ids is None else resident_ids
        resident_indices = torch.as_tensor(
            resident_ids, dtype=torch.long, device=self.device
        )
        self.lim_adapter.reset_requests(resident_indices)

    def copy_sfa_attention(self, *args, **kwargs):
        raise RuntimeError(
            "C8 full-topk mode uses HBM sparse attention for anchors and "
            "rolling QSFA for shared layers; CopySFA is not enabled."
        )

    def copy_shared_sfa_attention(
        self, layer, query_nope, query_rope, plan, current_inputs, forward_batch
    ):
        if self.shared_kv_prefetch is None:
            raise RuntimeError("C8 shared prefetch is not initialized.")
        return self.shared_kv_prefetch.attention(
            layer.layer_id - self.start_layer,
            query_nope,
            query_rope,
            layer,
            forward_batch,
            plan,
            current_inputs,
        )

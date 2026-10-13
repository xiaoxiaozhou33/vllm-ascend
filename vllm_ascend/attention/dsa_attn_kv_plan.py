# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Centralized DeepSeek-V4 attention-KV execution choices."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import os

import torch
import torch_npu
from vllm.triton_utils import tl, triton
from vllm.logger import logger

from vllm_ascend.attention.mixed_quant_sparse_flash_mla import (
    mixed_quant_sparse_flash_mla,
    mixed_quant_sparse_flash_mla_metadata,
)
from vllm_ascend.attention.sparse_flash_mla import sparse_flash_mla, sparse_flash_mla_metadata
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num
from vllm_ascend.quantization.methods.kv_cache.turboquant import is_turboquant

_BF16_KV_CACHE_DTYPES = frozenset({"bfloat16", "bf16"})


def _supports_dsv4_compressed_cache() -> bool:
    return get_current_hardware_profile().supports(HardwareCapability.DSV4_COMPRESSED_CACHE)


def resolve_dsv4_cache_dtype(cache_dtype, model_dtype: str) -> str:
    """Return the KV cache dtype the platform should pin for DeepSeek-V4.

    On A5 the launch request has to stay readable afterwards, because it is the
    only thing that separates an explicit bfloat16 KV request from ``auto``.
    ``auto`` and the model dtype resolve identically everywhere downstream, so
    collapsing every non-bfloat16 request to ``auto`` preserves the upstream
    values while keeping the mode recoverable.
    """
    if not _supports_dsv4_compressed_cache():
        return model_dtype
    return "bfloat16" if str(cache_dtype).lower() in _BF16_KV_CACHE_DTYPES else "auto"


def is_a5_bf16_kv_enabled(vllm_config) -> bool:
    """Return whether BF16 SparseFlashMla KV is enabled on A5.

    Callers must pass the engine ``vllm_config``. Do not look it up from the
    process-global current config: that context is only set during
    ``load_model()`` and a missing lookup would silently pick the FP8 plan.
    """
    if not _supports_dsv4_compressed_cache():
        return False
    cache_config = getattr(vllm_config, "cache_config", None)
    if cache_config is None:
        return False
    return str(cache_config.cache_dtype).lower() in _BF16_KV_CACHE_DTYPES


DSA_COMPRESSOR_SLOT_MAPPING_FLAT = 1
DSA_COMPRESSOR_SLOT_MAPPING_BLOCK_OFFSET = 2


@triton.jit(do_not_specialize=["tokens", "width"])
def _write_dsa_cache(
    cache,
    updates,
    slots,
    tokens,
    width,
    pages: tl.constexpr,
    page_size: tl.constexpr,
    page_stride: tl.constexpr,
    row_stride: tl.constexpr,
    update_stride: tl.constexpr,
    slot_stride: tl.constexpr,
    slot_column_stride: tl.constexpr,
    paired: tl.constexpr,
    columns: tl.constexpr,
):
    columns_idx = tl.arange(0, columns)
    for token in range(tl.program_id(0), tokens, tl.num_programs(0)):
        first = tl.load(slots + token * slot_stride).to(tl.int64)
        if paired:
            block = first
            offset = tl.load(slots + token * slot_stride + slot_column_stride).to(tl.int64)
        else:
            block = first // page_size
            offset = first % page_size
        if (first >= 0) & (block < pages) & (offset >= 0) & (offset < page_size):
            value = tl.load(updates + token * update_stride + columns_idx, columns_idx < width, other=0)
            address = block * page_stride + offset * row_stride + columns_idx
            tl.store(cache + address, value, columns_idx < width)


def write_dsa_cache(cache: torch.Tensor, updates: torch.Tensor, slots: torch.Tensor) -> None:
    """Write valid DSA slots while preserving packed physical-page strides."""
    if cache.ndim != 4 or cache.shape[2] != 1 or cache.stride(-1) != 1:
        raise ValueError("Cache must be [pages, page_size, 1, width] with contiguous rows")
    if updates.ndim not in (2, 3) or (updates.ndim == 3 and updates.shape[1] != 1):
        raise ValueError("Updates must be [tokens, width] or [tokens, 1, width]")
    if updates.shape[-1] != cache.shape[-1] or updates.stride(-1) != 1 or updates.dtype != cache.dtype:
        raise ValueError("Update rows must match the cache width and dtype")
    if slots.ndim not in (1, 2) or (slots.ndim == 2 and slots.shape[1] != 2):
        raise ValueError("Slots must be flat or [tokens, 2] block/offset pairs")
    if slots.shape[0] != updates.shape[0] or slots.dtype not in (torch.int32, torch.int64):
        raise ValueError("Slots must have one int32/int64 entry per update row")
    if not (cache.device == updates.device == slots.device):
        raise ValueError("Cache, updates and slots must be on the same device")
    tokens = updates.shape[0]
    if tokens == 0:
        return
    _write_dsa_cache[(min(get_vectorcore_num(), tokens),)](
        cache,
        updates,
        slots,
        tokens,
        cache.shape[-1],
        cache.shape[0],
        cache.shape[1],
        cache.stride(0),
        cache.stride(1),
        updates.stride(0),
        slots.stride(0),
        slots.stride(1) if slots.ndim == 2 else 0,
        slots.ndim == 2,
        triton.next_power_of_2(cache.shape[-1]),
        multibuffer=False,
    )


@dataclass(frozen=True)
class DsaAttnKvPlan:
    """The attention-KV plan only; indexer KV remains independently quantized."""

    uses_sparse_flash_mla: bool
    uses_kv_compress_epilog: bool
    layout_kv: str
    compressor_slot_mapping_format: int
    requires_block_offset_slots: bool
    sparse_attn_op: Callable[..., Any]
    sparse_attn_metadata_op: Callable[..., Any]
    sparse_attn_base_kwargs: dict[str, Any]
    sparse_attn_metadata_kwargs: dict[str, Any]
    include_metadata_device: bool
    applies_sparse_attn_runtime_kwargs: bool

    def get_dsa_sparse_attn_metadata_op(self):
        op = self.sparse_attn_metadata_op
        if os.getenv("VLLM_ASCEND_DEBUG_ATTN_KWARGS") and not globals().get("_MD_KWARGS_PROBED", False):

            def _md_probe(*args, **kwargs):
                def render(v):
                    if isinstance(v, torch.Tensor):
                        vals = v.detach().cpu().tolist() if v.numel() <= 48 else "..."
                        return f"Tensor{tuple(v.shape)}:{v.dtype}:{vals}"
                    return repr(v)

                globals()["_MD_KWARGS_PROBED"] = True
                logger.warning(
                    "[attn_md_kwargs] op=%s %s",
                    getattr(op, "__name__", repr(op)),
                    " ".join(f"{k}={render(v)}" for k, v in kwargs.items()),
                )
                return op(*args, **kwargs)

            return _md_probe
        return op

    def get_dsa_sparse_attn_metadata_kwargs(self, device) -> dict[str, Any]:
        kwargs = dict(self.sparse_attn_metadata_kwargs)
        if self.include_metadata_device:
            kwargs["device"] = str(device)
        return kwargs

    def get_dsa_sparse_attn_op(self):
        return self.sparse_attn_op

    def get_dsa_sparse_attn_base_kwargs(self) -> dict[str, Any]:
        return dict(self.sparse_attn_base_kwargs)

    def add_dsa_sparse_attn_extra_kwargs(self, extra_kwargs: dict[str, Any], **kwargs_to_add) -> None:
        if self.applies_sparse_attn_runtime_kwargs:
            extra_kwargs.update(kwargs_to_add)

    def get_dsa_compressor_slot_mapping_format(self) -> int:
        return self.compressor_slot_mapping_format

    def format_dsa_slot_mapping(self, slot_mapping: torch.Tensor, block_size: int | torch.Tensor) -> torch.Tensor:
        if not self.requires_block_offset_slots:
            return slot_mapping
        valid = slot_mapping >= 0
        invalid = torch.full_like(slot_mapping, -1)
        block_idx = torch.where(valid, torch.div(slot_mapping, block_size, rounding_mode="floor"), invalid)
        offset = torch.where(valid, slot_mapping % block_size, invalid)
        return torch.stack([block_idx, offset], dim=-1).to(torch.int32)

    def dsa_kv_compress_scatter(self, cache: torch.Tensor, x: torch.Tensor | None, slot_mapping: torch.Tensor) -> None:
        if x is None:
            return
        if self.uses_sparse_flash_mla:
            if slot_mapping.ndim != 1:
                raise ValueError(f"BF16 DSA slot_mapping must be [num_tokens], got {tuple(slot_mapping.shape)}.")
            if cache.dim() == 4 and cache.shape[2] == 1 and cache.stride(-1) == 1 and not cache.is_contiguous():
                # Padded pages (hybrid planner common_page_size): flatten would
                # silently materialize a copy and the scatter below would write
                # to it. Route through the stride-aware 2-level scatter shared
                # with the epilog branch instead. [-1, -1] rows are skipped.
                page = cache.shape[1]
                valid = slot_mapping >= 0
                skip = torch.full_like(slot_mapping, -1)
                blocks = torch.where(valid, torch.div(slot_mapping, page, rounding_mode="floor"), skip)
                offsets = torch.where(valid, slot_mapping % page, skip)
                indices = torch.stack([blocks, offsets], dim=-1).to(torch.int32)
                updates = x.view(x.shape[0], -1).to(cache.dtype).contiguous()
                torch_npu.npu_scatter_nd_update_(cache.squeeze(-2), indices, updates)
                return
            # Flatten the paged cache and keep a static [T, 1] index tensor so
            # ACLGraph capture matches the A5 FP8 / SFA one-dimensional slot
            # convention. Do not clamp PAD_SLOT_ID (-1) to 0: that overwrites
            # a live physical slot. The scatter kernel skips negative indices.
            flat_cache = cache.flatten(end_dim=1)
            indices = slot_mapping.view(-1, 1)
            updates = x.reshape((slot_mapping.shape[0],) + tuple(flat_cache.shape[1:]))
            torch_npu.npu_scatter_nd_update_(flat_cache, indices, updates)
            return
        if not self.uses_kv_compress_epilog:
            torch.ops._C_ascend.npu_scatter_nd_update_sk(cache, slot_mapping, x)
            return
        if os.getenv("VLLM_ASCEND_DEBUG_KV_GEOM"):
            # First-call geometry probe for the view-size wall: shape/stride of
            # the per-layer cache handed to the epilog scatter.
            if not getattr(DsaAttnKvPlan.dsa_kv_compress_scatter, "_probed", False):
                DsaAttnKvPlan.dsa_kv_compress_scatter._probed = True
                logger.warning(
                    "[kv_scatter] cache shape=%s stride=%s dtype=%s contig=%s "
                    "x shape=%s slot shape=%s",
                    tuple(cache.shape),
                    tuple(cache.stride()),
                    cache.dtype,
                    cache.is_contiguous(),
                    tuple(x.shape),
                    tuple(slot_mapping.shape),
                )
        # The DSV4 shared-tuple planner exposes each layer as a contiguous
        # region whose per-block stride still exceeds the dense page (page
        # rows padded to the group's largest member), so the flat view below
        # is not expressible. When no dtype conversion is involved the epilog
        # is a pure scatter: rewrite it through torch_npu.npu_scatter_nd_update_
        # with (block, offset) coordinates, which is stride-aware and skips
        # [-1, -1] rows (verified against a padded as_strided cache; the
        # aclnnScatterNdUpdateSk op dsa_v41 uses is not in this CANN's
        # libopapi.so). Boolean-mask indexing must not be used here: it lowers
        # to aclnnNonzeroV2, which crashes aicpu.
        if (
            cache.dtype == x.dtype
            and cache.stride(-1) == 1
            and cache.shape[2] == 1
            and slot_mapping.ndim == 1
        ):
            page = cache.shape[1]
            valid = slot_mapping >= 0
            skip = torch.full_like(slot_mapping, -1)
            blocks = torch.where(valid, torch.div(slot_mapping, page, rounding_mode="floor"), skip)
            offsets = torch.where(valid, slot_mapping % page, skip)
            indices = torch.stack([blocks, offsets], dim=-1).to(torch.int32)
            updates = x.view(x.shape[0], -1).to(cache.dtype).contiguous()
            torch_npu.npu_scatter_nd_update_(cache.squeeze(-2), indices, updates)
            return
        if (
            not cache.is_contiguous()
            and cache.dim() == 4
            and cache.shape[2] == 1
            and cache.stride(1) == cache.shape[-1]
            and cache.stride(-1) == 1
            and slot_mapping.ndim == 1
        ):
            # Quantizing epilog on padded pages: the flat view is not
            # expressible, but each single block is a dense 128-row subview.
            # Group the rows by block (argsort + host-side run split — no
            # nonzero/boolean-mask indexing, those crash aicpu) and re-issue
            # the epilog op per touched block with block-local slots. The op
            # semantics are untouched; only the geometry packaging changes.
            page = cache.shape[1]
            n_blocks = cache.shape[0]
            big = torch.iinfo(slot_mapping.dtype).max
            slots = torch.where(slot_mapping >= 0, slot_mapping, torch.full_like(slot_mapping, big))
            order = torch.argsort(slots)
            slots_sorted = slots.index_select(0, order)
            blk_sorted = torch.div(slots_sorted, page, rounding_mode="floor")
            blk_list = blk_sorted.tolist()
            x_rows = x.view(x.shape[0], -1)
            i = 0
            n = len(blk_list)
            while i < n and blk_list[i] < n_blocks:
                b = blk_list[i]
                j = i + 1
                while j < n and blk_list[j] == b:
                    j += 1
                sel = order[i:j]
                local = slots_sorted[i:j] - b * page
                torch.ops._C_ascend.kv_compress_epilog(
                    kv_compress_cache=cache[b].view(-1, 1, cache.shape[-1]),
                    x=x_rows.index_select(0, sel),
                    slot_mapping=local,
                    quant_group_size=64,
                    quant_mode=2,
                    round_scale_flag=True,
                    layout=1,
                )
                i = j
            return
        torch.ops._C_ascend.kv_compress_epilog(
            kv_compress_cache=cache.view(-1, 1, cache.shape[-1]),
            x=x.view(-1, x.shape[-1]),
            slot_mapping=slot_mapping,
            quant_group_size=64,
            quant_mode=2,
            round_scale_flag=True,
            layout=1,
        )


def get_dsa_attn_kv_plan(vllm_config, compress_ratio: int = 1) -> DsaAttnKvPlan:
    """Select a cache plan by hardware, dtype and the layer compression ratio."""
    if is_turboquant(vllm_config) and compress_ratio == 4:
        return DsaAttnKvPlan(
            uses_sparse_flash_mla=False,
            uses_kv_compress_epilog=False,
            layout_kv="PA_BBND",
            compressor_slot_mapping_format=DSA_COMPRESSOR_SLOT_MAPPING_BLOCK_OFFSET,
            requires_block_offset_slots=True,
            sparse_attn_op=mixed_quant_sparse_flash_mla,
            sparse_attn_metadata_op=mixed_quant_sparse_flash_mla_metadata,
            sparse_attn_base_kwargs={},
            sparse_attn_metadata_kwargs={},
            include_metadata_device=False,
            applies_sparse_attn_runtime_kwargs=True,
        )
    if not _supports_dsv4_compressed_cache():
        return DsaAttnKvPlan(
            uses_sparse_flash_mla=False,
            uses_kv_compress_epilog=False,
            layout_kv="PA_ND",
            compressor_slot_mapping_format=DSA_COMPRESSOR_SLOT_MAPPING_BLOCK_OFFSET,
            requires_block_offset_slots=True,
            sparse_attn_op=torch.ops._C_ascend.npu_sparse_attn_sharedkv,
            sparse_attn_metadata_op=torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata,
            sparse_attn_base_kwargs={},
            sparse_attn_metadata_kwargs={},
            include_metadata_device=True,
            applies_sparse_attn_runtime_kwargs=True,
        )

    use_bf16 = is_a5_bf16_kv_enabled(vllm_config)
    if use_bf16:
        return DsaAttnKvPlan(
            uses_sparse_flash_mla=True,
            uses_kv_compress_epilog=False,
            layout_kv="PA_BBND",
            compressor_slot_mapping_format=DSA_COMPRESSOR_SLOT_MAPPING_FLAT,
            requires_block_offset_slots=False,
            sparse_attn_op=sparse_flash_mla,
            sparse_attn_metadata_op=sparse_flash_mla_metadata,
            sparse_attn_base_kwargs={},
            sparse_attn_metadata_kwargs={},
            include_metadata_device=True,
            applies_sparse_attn_runtime_kwargs=True,
        )
    return DsaAttnKvPlan(
        uses_sparse_flash_mla=False,
        uses_kv_compress_epilog=True,
        layout_kv="PA_ND",
        compressor_slot_mapping_format=DSA_COMPRESSOR_SLOT_MAPPING_FLAT,
        requires_block_offset_slots=False,
        sparse_attn_op=torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv,
        sparse_attn_metadata_op=torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv_metadata,
        sparse_attn_base_kwargs={"kv_quant_mode": 1, "tile_size": 64, "rope_head_dim": 64},
        sparse_attn_metadata_kwargs={"kv_quant_mode": 1},
        include_metadata_device=False,
        applies_sparse_attn_runtime_kwargs=False,
    )

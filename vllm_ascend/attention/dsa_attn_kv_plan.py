# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Centralized DeepSeek-V4 attention-KV execution choices."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch_npu

from vllm_ascend.attention.sparse_flash_mla import sparse_flash_mla, sparse_flash_mla_metadata
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

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

# TEMP DEBUG: signatures already printed by VLLM_ASCEND_DEBUG_KV_SCATTER.
_DBG_SEEN_SIGS: set = set()


@dataclass(frozen=True)
class DsaAttnKvPlan:
    """The attention-KV plan only; indexer KV remains independently FP8."""

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
        return self.sparse_attn_metadata_op

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

    def format_dsa_slot_mapping(self, slot_mapping: torch.Tensor, block_size: int) -> torch.Tensor:
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
        # TEMP DEBUG (VLLM_ASCEND_DEBUG_KV_SCATTER=1): capture the live cache
        # geometry that feeds the view() calls below. Prints once per unique
        # (shape, stride) signature; on failure prints everything.
        import os as _os
        if _os.getenv("VLLM_ASCEND_DEBUG_KV_SCATTER", "0") == "1":
            sig = (tuple(cache.shape), tuple(cache.stride()), cache.storage_offset())
            if sig not in _DBG_SEEN_SIGS:
                _DBG_SEEN_SIGS.add(sig)
                print(
                    f"[kv-scatter-dbg] plan(sfm={self.uses_sparse_flash_mla}, "
                    f"epilog={self.uses_kv_compress_epilog}, layout={self.layout_kv}) "
                    f"cache{tuple(cache.shape)} stride{tuple(cache.stride())} "
                    f"off={cache.storage_offset()} dtype={cache.dtype} "
                    f"contig={cache.is_contiguous()} | x{tuple(x.shape)} "
                    f"xstride{tuple(x.stride())} | slot{tuple(slot_mapping.shape)}",
                    flush=True,
                )
            try:
                self._dsa_kv_compress_scatter_impl(cache, x, slot_mapping)
            except RuntimeError:
                print(
                    f"[kv-scatter-dbg] FAILED cache{tuple(cache.shape)} "
                    f"stride{tuple(cache.stride())} off={cache.storage_offset()} "
                    f"dtype={cache.dtype} contig={cache.is_contiguous()} | "
                    f"x{tuple(x.shape)} xstride{tuple(x.stride())} | "
                    f"slot{tuple(slot_mapping.shape)}",
                    flush=True,
                )
                raise
            return
        self._dsa_kv_compress_scatter_impl(cache, x, slot_mapping)

    def _dsa_kv_compress_scatter_impl(self, cache: torch.Tensor, x: torch.Tensor | None, slot_mapping: torch.Tensor) -> None:
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
        # The 0.29 hybrid planner pads every page in a mixed group to the
        # largest member's page size (attn_utils common_page_size), so the
        # paged cache stride exceeds the dense page and the flat view below
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
        torch.ops._C_ascend.kv_compress_epilog(
            kv_compress_cache=cache.view(-1, 1, cache.shape[-1]),
            x=x.view(-1, x.shape[-1]),
            slot_mapping=slot_mapping,
            quant_group_size=64,
            quant_mode=2,
            round_scale_flag=True,
            layout=1,
        )


def get_dsa_attn_kv_plan(vllm_config) -> DsaAttnKvPlan:
    """Return the explicit A5 BF16 or upstream-compatible FP8 DSA plan."""
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

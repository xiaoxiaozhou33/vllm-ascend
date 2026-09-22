#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Weight-layout handling for the static/rotating expert split.

All device-side expert tensors share one invariant: the expert dimension is
dim 0 and it is the outermost dimension of the underlying storage, so the
first ``S`` experts are a contiguous storage prefix and the last ``R`` experts
a contiguous storage suffix. That holds for:

  - ND dense tensors (unquantized bf16, fp32 scales, flattened scale views)
  - FRACTAL_NZ (29) int8 weights (W8A8 dynamic)
  - FRACTAL_NZ cast-with-customize-dtype mxfp4 packed weights (W4A8_MXFP)

torch_npu's ``copy_`` refuses tensors carrying an internal format, so every
host<->device movement of NZ weights goes through ``untyped_storage()`` slices,
which strip the format tag and degrade to a plain byte memcpy. NZ weights must
NEVER be made ``.contiguous()`` here - that would trigger a format-aware
re-layout and silently corrupt the physical bytes.

Two subtle rules for NZ tensors whose kernel-facing form is a ``transpose(1,2)``
view (the W4A8_MXFP weights): the physical NZ layout is decided by the shape
present at cast time, i.e. the PRE-transpose shape. A rebuilt tensor must
therefore be allocated with the pre-view shape, NZ-cast, byte-filled, and only
then transposed into the kernel-facing view.

Non-contiguous *scale* views (e.g. the W4A8_MXFP transposed 4D scale view) are
materialized with ``.contiguous()`` before splitting: the kernel reads them
through the view strides, and a dense materialization preserves the logical
values while giving one canonical byte layout on both the shrunk device side
and the rotating slot side.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch_npu

from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

_ND_FORMATS = (0, 2)  # ACL_FORMAT_UNDEFINED / ACL_FORMAT_ND


def npu_format_of(t: torch.Tensor) -> int:
    try:
        return int(torch_npu.get_npu_format(t))
    except Exception:
        return 0


def is_nz_tensor(t: torch.Tensor) -> bool:
    return npu_format_of(t) not in _ND_FORMATS


def bytes_per_expert(t: torch.Tensor) -> int:
    """Storage bytes occupied by one expert row; dim 0 must be storage-major."""
    num_experts = t.shape[0]
    total = t.untyped_storage().nbytes()
    if num_experts == 0 or total % num_experts != 0:
        raise ValueError(
            f"expert tensor '{tuple(t.shape)}' storage ({total}B) is not divisible by "
            f"num_experts={num_experts}; this quantization layout is not supported by "
            "expert rotation yet."
        )
    return total // num_experts


def copy_storage_slice(dst: torch.Tensor, src_storage: torch.UntypedStorage, non_blocking: bool = False) -> None:
    """Byte-level copy of ``src_storage`` into ``dst``'s storage (format-agnostic).

    The destination is addressed as a storage SLICE (``[0:nbytes]``): slicing
    detaches the slice from the allocation's internal-format tag, which makes
    torch_npu degrade the copy to a plain aclrtMemcpy instead of the aclnn
    format-aware operator that rejects internal formats. Copying against the
    whole storage object of an NZ tensor raises
    ``copy_ do not support internal format`` on strict torch_npu builds.
    """
    dst.untyped_storage()[0 : src_storage.nbytes()].copy_(src_storage, non_blocking=non_blocking)


def absorb_suffix_to_pinned(t: torch.Tensor, num_rotating: int) -> torch.Tensor:
    """D2H the last ``num_rotating`` expert rows into a fresh pinned uint8 buffer."""
    num_experts = t.shape[0]
    per_e = bytes_per_expert(t)
    st = t.untyped_storage()
    suffix = st[(num_experts - num_rotating) * per_e : num_experts * per_e]
    pinned = torch.empty(suffix.nbytes(), dtype=torch.uint8, pin_memory=True)
    pinned.untyped_storage().copy_(suffix)
    return pinned


def _replace_layer_tensor(layer: torch.nn.Module, attr: str, new_tensor: torch.Tensor) -> None:
    cur = getattr(layer, attr)
    if isinstance(cur, torch.nn.Parameter):
        cur.data = new_tensor
    else:
        setattr(layer, attr, new_tensor)


def _nz_cast_like(quant_type: QuantType, new: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Cast a fresh allocation to the same internal format as ``ref``.

    The cast must happen while the allocation's pointer is still private (no
    view has been handed out), mirroring what process_weights_after_loading
    does for the original tensor.
    """
    if quant_type == QuantType.W4A8MXFP:
        return torch_npu.npu_format_cast(
            new, 29, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
        )
    if quant_type == QuantType.W8A8:
        return torch_npu.npu_format_cast(new, ACL_FORMAT_FRACTAL_NZ)
    # Unquantized weights: maybe_trans_nz may or may not have cast them.
    return torch_npu.npu_format_cast(new, npu_format_of(ref))


def _rebuild_expert_rows(
    t: torch.Tensor,
    num_rows: int,
    *,
    nz: bool,
    transposed_view: bool,
    quant_type: QuantType,
) -> torch.Tensor:
    """Build a standalone device tensor holding the first ``num_rows`` expert rows of ``t``.

    Mirrors the exact construction of the source tensor: fresh empty allocation
    (NZ-cast while the pointer is still private) -> byte-level D2D prefix copy
    -> identical view transform.

    ``transposed_view`` marks tensors whose kernel-facing form is
    ``t == base.transpose(1, 2)`` over the cast storage. The physical NZ layout
    is decided by the shape at cast time, so the rebuild allocates with the
    PRE-view shape, casts, byte-fills, and only then transposes.
    """
    if nz:
        per_e = bytes_per_expert(t)
        alloc_shape = (num_rows, t.shape[2], t.shape[1]) if transposed_view else (num_rows,) + tuple(t.shape[1:])
        new = torch.empty(alloc_shape, dtype=t.dtype, device=t.device)
        new = _nz_cast_like(quant_type, new, t)
        # NPU->NPU storage copy_ still trips the aclnn internal-format check on
        # this device generation, so the D2D prefix copy is staged through
        # pinned host memory (D2H + H2D both bypass the check). Load-time only.
        nbytes = num_rows * per_e
        staging = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        staging.untyped_storage().copy_(t.untyped_storage()[:nbytes])
        copy_storage_slice(new, staging.untyped_storage())
        del staging
        if transposed_view:
            new = new.transpose(1, 2)
        return new
    # ND dense path
    if not t.is_contiguous():
        t = t.contiguous()
    return t[:num_rows].clone()


@dataclass
class RotatingComponent:
    """One weight component (e.g. w13) of one MoE layer after splitting."""

    role: str  # key in MoEWeights: w1 / w2 / w1_scale / w2_scale
    attr: str  # attribute name on the RoutedExperts module
    pinned: torch.Tensor  # pinned uint8 buffer holding R experts of device-format bytes
    slots: list[torch.Tensor]  # HBM rotation slots (shared across layers), same view form
    nbytes: int


@dataclass(frozen=True)
class _RoleSpec:
    attrs: tuple[str, ...]  # layer attr candidates, first hit wins
    transposed_view: bool = False  # NZ weight consumed as a transpose(1,2) view
    materialize_view: bool = False  # non-contiguous ND view; densify first


_ROLE_SPECS: dict[tuple[QuantType, str], _RoleSpec] = {
    (QuantType.W4A8MXFP, "w1"): _RoleSpec(attrs=("w13_weight_packed", "w13_weight"), transposed_view=True),
    (QuantType.W4A8MXFP, "w2"): _RoleSpec(attrs=("w2_weight_packed", "w2_weight"), transposed_view=True),
    (QuantType.W4A8MXFP, "w1_scale"): _RoleSpec(attrs=("w13_weight_scale",), materialize_view=True),
    (QuantType.W4A8MXFP, "w2_scale"): _RoleSpec(attrs=("w2_weight_scale",), materialize_view=True),
    (QuantType.W8A8, "w1"): _RoleSpec(attrs=("w13_weight",)),
    (QuantType.W8A8, "w2"): _RoleSpec(attrs=("w2_weight",)),
    (QuantType.W8A8, "w1_scale"): _RoleSpec(attrs=("w13_weight_scale_fp32",)),
    (QuantType.W8A8, "w2_scale"): _RoleSpec(attrs=("w2_weight_scale",)),
    (QuantType.NONE, "w1"): _RoleSpec(attrs=("w13_weight",)),
    (QuantType.NONE, "w2"): _RoleSpec(attrs=("w2_weight",)),
}

SUPPORTED_QUANT_TYPES = tuple(dict.fromkeys(qt for qt, _ in _ROLE_SPECS))

# Roles required for every supported quant type; others (scales) are optional.
_REQUIRED_ROLES = ("w1", "w2")


def supported_quant_type(quant_type: QuantType) -> bool:
    return any(qt == quant_type for qt, _ in _ROLE_SPECS)


def split_layer_weights(
    layer: torch.nn.Module,
    quant_type: QuantType,
    num_rotating: int,
    *,
    num_slots: int = 0,
    verify: bool = False,
) -> dict[str, RotatingComponent]:
    """Split one MoE layer's weights into a static prefix + rotating suffix.

    Called right after the quant method's ``process_weights_after_loading`` so
    every tensor is already in its final device format. The layer's attributes
    are shrunk in place to the first ``E - R`` experts; the suffix bytes are
    absorbed into a pinned host buffer and a same-form slot tensor is created.
    """
    if not supported_quant_type(quant_type):
        raise ValueError(
            f"expert rotation does not support quant_type={quant_type} yet "
            f"(supported: {[qt.name for qt in SUPPORTED_QUANT_TYPES]})"
        )
    components: dict[str, RotatingComponent] = {}
    for (qt, role), spec in _ROLE_SPECS.items():
        if qt != quant_type:
            continue
        attr = next((a for a in spec.attrs if getattr(layer, a, None) is not None), None)
        if attr is None:
            if role in _REQUIRED_ROLES:
                raise ValueError(f"expert rotation: layer has none of {spec.attrs}")
            continue  # optional component (e.g. scales for unquantized)
        t = getattr(layer, attr)
        if isinstance(t, torch.nn.Parameter):
            t = t.data
        if spec.materialize_view and not t.is_contiguous():
            # canonical dense layout of the view; logical values unchanged
            _replace_layer_tensor(layer, attr, t.contiguous())
            t = getattr(layer, attr)
            if isinstance(t, torch.nn.Parameter):
                t = t.data
        num_experts = t.shape[0]
        num_static = num_experts - num_rotating
        if num_static <= 0:
            raise ValueError(
                f"num_rotating_experts={num_rotating} must be smaller than the local "
                f"expert count {num_experts} (attr {attr})"
            )
        nz = is_nz_tensor(t)
        pinned = absorb_suffix_to_pinned(t, num_rotating)
        import os as _os

        if _os.getenv("VLLM_ASCEND_ER_NO_SHRINK"):
            # EXPERIMENT: keep the original full-size weights on device; the
            # runtime split will narrow dim0 views instead of using rebuilt
            # tensors. Isolates "rebuilt tensor" vs "two-call slicing".
            static = t
        else:
            static = _rebuild_expert_rows(
                t, num_static, nz=nz, transposed_view=spec.transposed_view, quant_type=quant_type
            )
        if verify:
            _verify_split_bytes(t, static, pinned, num_static)
        # Slots are shared across ALL layers (same R, same shapes); only the first
        # split layer allocates them. Slot contents are don't-care (overwritten by
        # the first fill); only the allocation + cast + view recipe matters.
        slots = [
            _rebuild_expert_rows(t, num_rotating, nz=nz, transposed_view=spec.transposed_view, quant_type=quant_type)
            for _ in range(num_slots)
        ]
        import os as _os2

        if not _os2.getenv("VLLM_ASCEND_ER_NO_SHRINK"):
            _replace_layer_tensor(layer, attr, static)
        # release the original full-size storage eagerly, layer by layer
        del t
        torch.npu.empty_cache()
        components[role] = RotatingComponent(
            role=role, attr=attr, pinned=pinned, slots=slots, nbytes=pinned.untyped_storage().nbytes()
        )
    return components


def _verify_split_bytes(t: torch.Tensor, static: torch.Tensor, pinned: torch.Tensor, num_static: int) -> None:
    """Load-time sanity: shrunk prefix + pinned suffix must reproduce the original bytes."""
    per_e = bytes_per_expert(t)
    num_experts = t.shape[0]
    full_n = t.untyped_storage().nbytes()
    host_full = torch.empty(full_n, dtype=torch.uint8, pin_memory=True)
    host_full.untyped_storage()[0:full_n].copy_(t.untyped_storage()[0:full_n])
    static_n = static.untyped_storage().nbytes()
    host_static = torch.empty(static_n, dtype=torch.uint8, pin_memory=True)
    host_static.untyped_storage()[0:static_n].copy_(static.untyped_storage()[0:static_n])
    ok_static = torch.equal(host_static, host_full[: num_static * per_e])
    ok_pinned = torch.equal(pinned, host_full[num_static * per_e : num_experts * per_e])
    del host_full, host_static
    if not (ok_static and ok_pinned):
        raise RuntimeError(
            f"expert rotation split verification failed (prefix_ok={ok_static}, suffix_ok={ok_pinned}); "
            "the quantization layout does not survive the byte-level split."
        )


def fill_slot_from_pinned(slot: torch.Tensor, pinned: torch.Tensor, stream: torch.npu.Stream) -> None:
    """One big-block H2D per component; the whole R-expert suffix is contiguous."""
    with torch.npu.stream(stream):
        copy_storage_slice(slot, pinned.untyped_storage(), non_blocking=True)

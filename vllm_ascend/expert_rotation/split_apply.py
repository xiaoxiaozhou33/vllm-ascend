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
"""Two-GMM split of the MoE MLP stage along the static/rotating expert boundary.

Why split HERE (post-dispatch, MLP entry) and not at the routing level:

Every dispatch implementation (MC2 ``npu_moe_distribute_dispatch``, AllToAll
``repeat_interleave`` reordering, AllGather ``npu_moe_init_routing``) delivers
tokens sorted by LOCAL expert id in ascending order, with ``group_list``
counting tokens per local expert. Local experts [0, S) are static and [S, E)
rotating, so the sorted rows split exactly once at

    n_s = sum(group_list[:S])          (count mode)
    n_s = group_list[S-1]              (cumsum mode)

- static segment: rows [0, n_s), group_list[:S], the layer's shrunk weights
- rotating segment: rows [n_s, T), rebased group_list, the slot views

Padding rows after the valid total land in the rotating segment and are
ignored exactly as the single-call kernel ignored them. Outputs concatenate
back into the original row layout, so dispatch/combine, EP ownership,
topk-weighted reduction (all applied by the combine stage from dispatch
metadata) stay untouched - no partial sums, no renormalize bookkeeping.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import torch

from vllm.logger import logger


def _slice_upto(t: torch.Tensor | None, n: int) -> torch.Tensor | None:
    return None if t is None else t[:n]


def _slice_from(t: torch.Tensor | None, n: int) -> torch.Tensor | None:
    return None if t is None else t[n:]


def rotation_split_apply(mlp_compute_input, ctx, apply_mlp: Callable):
    """Run the MLP stage as static + rotating GMM calls.

    ``apply_mlp`` is ``MoECommMethod._apply_mlp`` (returns ``(output, event)``).
    ``ctx`` is the :class:`RotationCtx` stashed on ``_EXTRA_CTX``.
    """
    S = ctx.num_static
    gl = mlp_compute_input.group_list
    glt = mlp_compute_input.group_list_type
    x = mlp_compute_input.hidden_states
    total_rows = x.shape[0]

    # MC2 global_bs mode: rows are sorted by GLOBAL expert id and this rank's
    # local block lives at row_offset; group_list positions are global. The
    # uniform (non-global) mode is row_offset=0.
    off = getattr(ctx, "row_offset", 0) or 0
    own = getattr(ctx, "own_rows", None)

    if glt == 1:  # per-expert counts (local)
        n_s = int(gl[:S].sum().item()) if S > 0 else 0
        n_rot = int(gl[S:].sum().item())
        static_gl = gl[:S]
        rot_gl = gl[S:]
    elif glt == 0:  # prefix sums; positions are GLOBAL when off > 0
        g_s = int(gl[S - 1].item()) if S > 0 else off
        g_e = int(gl[-1].item())
        n_s = g_s - off
        n_rot = g_e - g_s
        static_gl = gl[:S] - off
        rot_gl = gl[S:] - g_s
        if own is not None and (g_e - off) > own:
            raise RuntimeError(
                f"[expert_rotation] own block mismatch: gl_end-off={g_e - off} own={own}"
            )
    else:
        raise NotImplementedError(f"expert rotation does not support group_list_type={glt}")

    import os as _os

    if _os.getenv("VLLM_ASCEND_ER_DEBUG"):
        ds = mlp_compute_input.dynamic_scale
        w1 = mlp_compute_input.weights.w1
        w1s = mlp_compute_input.weights.w1_scale
        try:
            # locate the own block: nonzero-row index range of x (padding rows
            # should be zero for dispatch outputs with capacity padding)
            with torch.no_grad():
                rowsum = x.float().abs().sum(dim=1) if x.dtype not in (torch.float32, torch.bfloat16, torch.float16) else x.abs().sum(dim=1)
                nz_idx = (rowsum > 0).nonzero().flatten()
                nz_first = int(nz_idx[0].item()) if nz_idx.numel() else -1
                nz_last = int(nz_idx[-1].item()) if nz_idx.numel() else -1
                nz_count = int(nz_idx.numel())
            with open("/tmp/er_dbg.log", "a") as _f:
                _f.write(
                    f"mid={ctx.moe_instance_id} glt={glt} gl={tuple(gl.shape)}/{gl.dtype} "
                    f"n_s={n_s} n_rot={n_rot} rows={total_rows} x={tuple(x.shape)}/{x.dtype} "
                    f"ds={'None' if ds is None else f'{tuple(ds.shape)}/{ds.dtype}'} "
                    f"w1={type(w1).__name__}"
                    f"{tuple(w1.shape) if hasattr(w1, 'shape') else [tuple(t.shape) for t in w1]} "
                    f"w1s={'None' if w1s is None else type(w1s).__name__ + str(tuple(w1s.shape) if hasattr(w1s, 'shape') else [tuple(t.shape) for t in w1s])} "
                    f"quant={mlp_compute_input.quant.quant_type} fusion={mlp_compute_input.fusion} "
                    f"act={mlp_compute_input.activation} "
                    f"glvals={gl[:6].tolist()}...{gl[-6:].tolist()} rots={rot_gl.tolist()} "
                    f"NZ[{nz_first},{nz_last}]cnt={nz_count} off={off} own={own}\n"
                )
        except Exception:
            pass

    if n_s < 0 or n_s > total_rows:
        raise RuntimeError(
            f"[expert_rotation] invalid split boundary n_s={n_s} (rows={total_rows}, S={S}); "
            "dispatch ordering assumption violated."
        )

    def _mlp_slice(rows_lo: int, rows_hi: int, group_list: torch.Tensor, weights=None):
        # Slices MUST be materialized contiguous: aclnn grouped-matmul variants
        # on this CANN build mishandle inputs whose storage_offset != 0 (they
        # read from the base pointer), which silently computes wrong rows. The
        # NZ-weight dim0 views are rejected outright (error 561103); fp8 x /
        # e8m0 scale offset views fail silently instead - same root cause.
        def _rows(t):
            return None if t is None else t[rows_lo:rows_hi].contiguous()

        return replace(
            mlp_compute_input,
            hidden_states=x[rows_lo:rows_hi].contiguous(),
            group_list=group_list.contiguous(),
            dynamic_scale=_rows(mlp_compute_input.dynamic_scale),
            topk_scales=_rows(mlp_compute_input.topk_scales),
            weights=weights if weights is not None else mlp_compute_input.weights,
            # LoRA routing metadata does not survive the split; LoRA is rejected
            # at config time, drop the fields defensively.
            lora_context=None,
            expanded_row_idx=None,
            topk_ids=None,
        )

    out_s = None
    evt = None
    import os as _os4

    static_weights = mlp_compute_input.weights
    if _os4.getenv("VLLM_ASCEND_ER_NO_SHRINK"):

        def _narrow(w):
            if w is None:
                return None
            if isinstance(w, list):
                return [t[:S] if isinstance(t, torch.Tensor) else t for t in w]
            return w[:S] if isinstance(w, torch.Tensor) else w

        from dataclasses import replace as _replace

        static_weights = _replace(
            mlp_compute_input.weights,
            w1=_narrow(mlp_compute_input.weights.w1),
            w2=_narrow(mlp_compute_input.weights.w2),
            w1_scale=_narrow(mlp_compute_input.weights.w1_scale),
            w2_scale=_narrow(mlp_compute_input.weights.w2_scale),
        )
    if n_s > 0:
        out_s, evt = apply_mlp(_mlp_slice(off, off + n_s, static_gl, weights=static_weights))
        if out_s.shape[0] != n_s:
            raise RuntimeError(
                f"[expert_rotation] static GMM returned {out_s.shape[0]} rows, expected {n_s}"
            )

    out_r = None
    import os as _os2

    if n_rot > 0:
        if False:
            # experiment: padding-only tail; replace the zero-group rotating GMM
            # with an explicit zero fill matching the static output dtype
            out_r = torch.zeros((total_rows - n_s,) + tuple(x.shape[1:]), dtype=out_s.dtype, device=x.device)
            ctx.manager.mark_slot_consumed(ctx)
        else:
            ctx.manager.wait_slot_ready(ctx)
            rot_weights = ctx.manager.build_rot_weights(mlp_compute_input.weights, ctx.slot_idx)
            out_r, _ = apply_mlp(_mlp_slice(off + n_s, off + n_s + n_rot, rot_gl, weights=rot_weights))
            ctx.manager.mark_slot_consumed(ctx)
        if out_r.shape[0] != n_rot:
            raise RuntimeError(
                f"[expert_rotation] rotating GMM returned {out_r.shape[0]} rows, expected {n_rot}"
            )

    import os as _os3

    if _os3.getenv("VLLM_ASCEND_ER_HASH"):
        with torch.no_grad():
            s32 = out_s.float() if out_s is not None else None
            r32 = out_r.float() if out_r is not None else None
        with open("/tmp/er_hash.log", "a") as _f:
            if s32 is not None:
                _f.write(
                    f"SPLIT-S r={_os3.getpid()} epr={getattr(ctx, 'ep_rank', -1)} mid={ctx.moe_instance_id} rows={out_s.shape[0]} "
                    f"msum={s32.sum().item():.4e} mabs={s32.abs().sum().item():.4e}\n"
                )
            if r32 is not None:
                _f.write(
                    f"SPLIT-R r={_os3.getpid()} epr={getattr(ctx, 'ep_rank', -1)} mid={ctx.moe_instance_id} rows={out_r.shape[0]} "
                    f"msum={r32.sum().item():.4e} mabs={r32.abs().sum().item():.4e}\n"
                )

    if off == 0 and out_s is not None and out_r is not None and out_s.shape[0] + out_r.shape[0] == total_rows:
        return torch.cat([out_s, out_r], dim=0), evt
    if off == 0 and out_s is not None and out_r is None and out_s.shape[0] == total_rows:
        return out_s, evt
    # global-layout mode: reassemble into a full-size buffer at the own offsets
    ref_t = out_s if out_s is not None else out_r
    out_dtype = ref_t.dtype if ref_t is not None else torch.bfloat16
    out = torch.zeros((total_rows,) + tuple(x.shape[1:]), dtype=out_dtype, device=x.device)
    if out_s is not None:
        out[off : off + n_s] = out_s
    if out_r is not None:
        out[off + n_s : off + n_s + n_rot] = out_r
    return out, evt

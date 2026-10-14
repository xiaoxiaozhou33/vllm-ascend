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
"""Runtime manager for the static/rotating expert split.

Ownership model
---------------
HBM:
  - static weights: each layer keeps its shrunk ``[E - R, ...]`` parameters,
    allocated once at load time, never moved again.
  - rotation slots: ``num_slots`` (2) slot tensors per weight component,
    shared by ALL MoE layers. Slot ``L % 2`` holds layer L's rotating experts
    while layer L computes; the copy stream fills slot ``(L+1) % 2`` with
    layer L+1's block concurrently (double buffering).

Host:
  - one pinned buffer per (layer, component) holding the device-format bytes
    of that layer's R rotating experts. Filling a slot is a single big-block
    H2D per component (never per-expert small copies - measured 2.9MB copies
    only sustain 20-47GB/s while large blocks reach ~100GB/s).

Event discipline (per slot)
---------------------------
  copy_done[s]: recorded on the copy stream after the H2D fill; the compute
      stream waits it before the rotating-segment GMM of the layer bound to s.
  slot_free[s]: recorded on the compute stream after that layer's rotating GMM
      has been enqueued; the copy stream waits it before overwriting s.
  => exactly one layer's block is in flight at any time; the host never
     floods the async copy queue (a known TTFT hazard in naive designs).

Every slot use (prefill fill or fallback sync fill) MUST end with a
``slot_free`` record; the events are pre-recorded once at init so the first
wait cannot deadlock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.logger import logger

from vllm_ascend import envs
from vllm_ascend.expert_rotation.config import ExpertRotationConfig
from vllm_ascend.expert_rotation.weight_layout import (
    RotatingComponent,
    bytes_per_expert,
    copy_storage_slice,
    split_layer_weight_lists,
    split_layer_weights,
    supported_quant_type,
)


@dataclass
class LayerRotationState:
    """Per-layer rotating bookkeeping (pinned blocks + sizes)."""

    moe_instance_id: int
    num_local_experts: int
    num_static: int
    num_rotating: int
    components: dict[str, RotatingComponent]
    pinned_bytes: int = 0
    # merged-GMM mode: resident static weights per role (device tensors, the
    # D2D source for the combined slot's prefix)
    static_srcs: dict[str, torch.Tensor] = field(default_factory=dict)
    # MegaMoe mode: the layer's shrunk per-expert static lists per role
    # (list entries are the resident static experts; the fused op receives
    # static entries + slot unbind(0) views at run time)
    mega_static_lists: dict[str, list[torch.Tensor]] = field(default_factory=dict)


@dataclass
class RotationCtx:
    """Per-forward handed to the MLP split point via ``_EXTRA_CTX.rotation_ctx``."""

    manager: "ExpertRotationManager"
    moe_instance_id: int
    slot_idx: int
    num_static: int
    num_rotating: int
    # filled lazily by mark_slot_consumed bookkeeping
    consumed: bool = False


@dataclass
class _SlotState:
    holder: int | None = None  # moe_instance_id whose block is loaded/loading
    copy_done: torch.npu.Event = field(default_factory=torch.npu.Event)
    slot_free: torch.npu.Event = field(default_factory=torch.npu.Event)
    d2d_done: torch.npu.Event = field(default_factory=torch.npu.Event)


class ExpertRotationManager:
    """Process-wide singleton owning slots, pinned blocks and the copy stream."""

    def __init__(self, config: ExpertRotationConfig):
        self.config = config
        self.num_slots = config.num_slots
        # merged-GMM mode: static (D2D) + rotating (H2D) experts gathered into
        # one combined S+R slot, single GMM per weight matrix at run time.
        self.merged_gmm = envs.VLLM_ASCEND_ER_MERGED_GMM
        # MegaMoe mode (enable_fused_mc2=2): weights split as per-expert LISTS
        # (the quant method disposed the raw tensors after building the
        # cann_mega_moe_* clones); decided at the first split_layer, which runs
        # under the same use_cann_megamoe() condition that built those lists.
        # In this mode slots stay R-row (no merged S+R capacity, no D2D
        # prefix) and the fused op receives static entries + slot unbind
        # views spliced into one per-expert list.
        self.mega_mode = False
        self.layers: dict[int, LayerRotationState] = {}
        # pre-sized mode: per-layer staging module + static->staging param map
        self._staging_modules: dict[int, torch.nn.Module] = {}
        self._staging_params: dict[int, dict[int, torch.nn.Parameter]] = {}
        self._orig_process_weights: dict[int, Any] = {}
        # slot views per MoEWeights role: role -> [slot0_view, slot1_view]
        self.slot_views: dict[str, list[torch.Tensor]] = {}
        self.slot_states: list[_SlotState] = [_SlotState() for _ in range(self.num_slots)]
        self.copy_stream = torch.npu.Stream() if config.prefetch_enabled else None
        # D2D gather stream for the merged mode's static prefix; runs the
        # device-side copy in parallel with the copy stream's H2D suffix fill
        # (different engines: HBM/vector vs PCIe/SDMA).
        self.d2d_stream = (
            torch.npu.Stream() if (config.prefetch_enabled and self.merged_gmm) else None
        )
        self._last_served: int | None = None
        self._registered = 0
        self._pinned_bytes_total = 0
        # record the freeing events once so the very first wait cannot hang
        for s in self.slot_states:
            s.slot_free.record()
        logger.info(
            "[expert_rotation] manager created: num_slots=%d prefetch=%s merged_gmm=%s",
            self.num_slots,
            config.prefetch_enabled,
            self.merged_gmm,
        )

    # ------------------------------------------------------------------
    # load time
    # ------------------------------------------------------------------

    def register_runner(self, runner) -> None:
        """Called from AscendMoERunner.__init__; validates the layer early."""
        import os as _os

        if _os.getenv("VLLM_ASCEND_ER_DEBUG"):
            em = getattr(runner, "_expert_map", None)
            if em is not None:
                with open("/tmp/er_emap.log", "a") as _f:
                    t = em if not hasattr(em, "cpu") else em.cpu()
                    _f.write(f"shape={tuple(t.shape)} head20={t.flatten()[:20].tolist()} nonneg={[i for i, v in enumerate(t.flatten()[:40].tolist()) if v >= 0]}\n")
        quant_type = runner.quant_type
        if not supported_quant_type(quant_type):
            raise ValueError(
                f"[expert_rotation] moe layer {runner.moe_instance_id} uses unsupported "
                f"quant_type={quant_type}; disable expert_rotation_config or use a "
                "supported quantization (NONE / W8A8 / W4A8MXFP)."
            )
        self._registered += 1

    # ------------------------------------------------------------------
    # pre-sized allocation (single-card / tight-HBM deployments)
    # ------------------------------------------------------------------

    def pre_create_layer(self, runner) -> None:
        """Called from AscendMoERunner.__init__ BEFORE checkpoint loading.

        Mirrors the reference offload bootstrap: wrap routed_experts.weight_loader
        (static experts pass through; rotating experts' shards are redirected,
        via an id-shifted orig_wl call, into an R-row staging param), then
        re-create the device weights at the STATIC size S so the params capture
        the wrapped loader and the full-size transient is skipped.
        """
        S = self._static_size(runner)
        R = self.config.num_rotating_experts
        runner_ref = runner
        re = runner.routed_experts
        orig_wl = re.weight_loader

        # global -> local map (forward) and local -> global (reverse), built
        # once from the routed_experts module's OWN map — the exact map
        # orig_wl applies internally (expert_map_manager), so the rev[]
        # lookup in _rotation_weight_loader stays its precise inverse.
        # Identity is only valid when experts are not EP-sharded; under
        # DP+EP a runner-level mirror attr may be absent, and an identity
        # fallback would mis-route foreign experts into staging (OOB on the
        # start=0 rank, silent skip/wrong-rows on the others).
        n_global = runner.moe_config.num_experts
        if runner.moe_config.num_local_experts == n_global:
            # ep_size == 1: identity mapping over the full expert set
            fwd = list(range(n_global))
        else:
            map_fn = getattr(
                re, "_map_global_expert_id_to_local_expert_id", None
            )
            if map_fn is None:
                raise RuntimeError(
                    "[expert_rotation] EP sharding active (num_local_experts="
                    f"{runner.moe_config.num_local_experts} < {n_global}) but "
                    "routed_experts exposes no global->local expert map; "
                    "refusing identity fallback"
                )
            fwd = [int(map_fn(g)) for g in range(n_global)]
        rev = {}
        for g, l in enumerate(fwd):
            if l >= 0 and l not in rev:
                rev[l] = g

        # Keep the UNWRAPPED process_weights: fused_moe installs the
        # rotation wrap right after pre_create_layer returns, and the staging
        # transform must not re-enter split_layer through that wrap.
        self._orig_process_weights[runner.moe_instance_id] = runner._quant_method.process_weights_after_loading
        staging_holder = torch.nn.Module()

        def _rotation_weight_loader(param, loaded_weight, weight_name, shard_id, expert_id, **kwargs):
            local = fwd[expert_id] if 0 <= expert_id < len(fwd) else -1
            if local >= S:
                stage = self._staging_params.get(runner_ref.moe_instance_id, {}).get(id(param))
                if stage is None:
                    # fallback: this param has no staging twin (unexpected);
                    # refuse rather than silently writing OOB rows.
                    raise RuntimeError(
                        f"[expert_rotation] pre_sized: no staging param for "
                        f"{weight_name} (shard {shard_id}, expert {expert_id})"
                    )
                shifted_global = rev.get(local - S)
                if shifted_global is None:
                    return None
                return orig_wl(stage, loaded_weight, weight_name, shard_id, shifted_global, **kwargs)
            return orig_wl(param, loaded_weight, weight_name, shard_id, expert_id, **kwargs)

        re.weight_loader = _rotation_weight_loader

        # Re-create the STATIC device weights at S with the wrapped loader
        # (params capture routed_experts.weight_loader at create time).
        self._quant_create(runner, re, S)
        # Staging params at R on a scratch module, allocated on CPU: staging
        # lives from here until split_layer (i.e. through the whole checkpoint
        # load), so keeping it on device would defeat the pre-sizing entirely
        # (sum over layers of S+R == full size). The wrapped loader writes
        # rotating shards here CPU->CPU; split_layer moves it to device just
        # for the format transform, then absorbs to pinned and frees.
        with torch.device("cpu"):
            self._quant_create(runner, staging_holder, R)
        self._staging_modules[runner.moe_instance_id] = staging_holder
        self._staging_params[runner.moe_instance_id] = {
            id(getattr(re, name)): getattr(staging_holder, name)
            for name in self._moe_param_names(re)
        }
        # drop the upstream full-size transient eagerly
        torch.npu.empty_cache()

    def _quant_create(self, runner, holder, num_experts: int) -> None:
        re = runner.routed_experts
        runner._quant_method.create_weights(
            layer=holder,
            num_experts=num_experts,
            hidden_size=runner.hidden_size,
            intermediate_size_per_partition=runner.moe_config.intermediate_size_per_partition,
            params_dtype=re.params_dtype,
            weight_loader=re.weight_loader,
        )

    @staticmethod
    def _moe_param_names(re) -> list[str]:
        names = []
        for w in ("w13_weight_packed", "w13_weight", "w2_weight_packed", "w2_weight"):
            if getattr(re, w, None) is not None:
                names.append(w)
        for s in (
            "w13_weight_scale",
            "w2_weight_scale",
            "w13_weight_scale_fp32",
            "w13_weight_offset",
            "w2_weight_offset",
            "w13_scale_bias",
            "w2_scale_bias",
        ):
            if getattr(re, s, None) is not None:
                names.append(s)
        return names

    def _static_size(self, runner) -> int:
        return runner.moe_config.num_local_experts - self.config.num_rotating_experts

    @staticmethod
    def _collect_static_srcs(layer: torch.nn.Module, quant_type) -> dict[str, torch.Tensor]:
        """Snapshot the resident (shrunk) static tensors per role after the split.

        Merged-GMM mode uses them as the D2D source filling the combined slot's
        static prefix at every rotation fill. Must run AFTER the layer attrs
        have been shrunk/replaced (split_layer_weights or the pre-sized
        transform), so the references are to the final S-row device tensors.
        """
        from vllm_ascend.expert_rotation.weight_layout import _ROLE_SPECS

        srcs: dict[str, torch.Tensor] = {}
        for (qt, role), spec in _ROLE_SPECS.items():
            if qt != quant_type:
                continue
            attr = next((a for a in spec.attrs if getattr(layer, a, None) is not None), None)
            if attr is None:
                continue
            t = getattr(layer, attr)
            if isinstance(t, torch.nn.Parameter):
                t = t.data
            srcs[role] = t
        return srcs

    def _mark_layer(self, runner, state: "LayerRotationState") -> None:
        """Stamp the FULL logical local expert count (static + rotating) on the
        routed-experts layer object. Routing/dispatch must cover all experts
        even though the layer's weights are shrunk to the static count — the
        quant scheme reads this to size group_list / active_expert_range (see
        the num_local_experts threading in w4a8_mxfp4.py's apply)."""
        runner.routed_experts._rotation_full_local = state.num_static + state.num_rotating

    def split_layer(self, runner) -> None:
        """Called right after the quant method's process_weights_after_loading."""
        mid = runner.moe_instance_id
        if mid in self.layers:
            return
        layer = runner.routed_experts
        num_rotating = self.config.num_rotating_experts
        first = not self.slot_views
        # MegaMoe: same condition the quant method's process_weights_after_
        # loading evaluated when it built (and only left) the per-expert
        # cann_mega_moe_* lists — under it the tensor split would read the
        # disposed raw tensors, so route to the list split instead.
        from vllm.config import get_current_vllm_config
        from vllm_ascend.ascend_forward_context import use_cann_megamoe

        if use_cann_megamoe(get_current_vllm_config()):
            self.mega_mode = True
            self._split_layer_mega(runner, mid, num_rotating, first)
            return
        if self.config.pre_sized and mid in self._staging_modules:
            self._split_pre_sized(runner, mid, num_rotating, first)
            return
        num_local = runner.moe_config.num_local_experts
        components = split_layer_weights(
            layer,
            runner.quant_type,
            num_rotating,
            num_slots=self.num_slots if first else 0,
            slot_capacity=(num_local if self.merged_gmm else None),
            verify=self.config.verify_split,
        )
        state = LayerRotationState(
            moe_instance_id=mid,
            num_local_experts=num_local,
            num_static=num_local - num_rotating,
            num_rotating=num_rotating,
            components=components,
            pinned_bytes=sum(c.nbytes for c in components.values()),
            static_srcs=self._collect_static_srcs(layer, runner.quant_type),
        )
        self.layers[mid] = state
        self._mark_layer(runner, state)
        self._pinned_bytes_total += state.pinned_bytes
        import os as _os

        if first:
            for role, comp in components.items():
                if _os.getenv("VLLM_ASCEND_ER_NO_SHRINK"):
                    # EXPERIMENT: slot view = dim0 tail view of the original weight
                    # (layer attr now holds the FULL tensor); matches the rebuild form.
                    full = getattr(layer, comp.attr)
                    if isinstance(full, torch.nn.Parameter):
                        full = full.data
                    self.slot_views[role] = [full[full.shape[0] - num_rotating :]]
                else:
                    self.slot_views[role] = comp.slots
            # sanity: slot form must match the layer's (shrunk) weight form
            self._log_summary()
        logger.info(
            "[expert_rotation] layer %d split: %d static + %d rotating experts, pinned %.2f GiB",
            mid,
            state.num_static,
            state.num_rotating,
            state.pinned_bytes / 1024**3,
        )

    def _split_pre_sized(self, runner, mid: int, num_rotating: int, first: bool) -> None:
        """Pre-sized variant: the layer is ALREADY S-sized and transformed; the
        staging module holds the R rotating experts (checkpoint-loaded, not yet
        transformed). Run the same process_weights transform on staging, absorb
        it into pinned, build the slots, then free the staging module."""
        staging = self._staging_modules.pop(mid, None)
        param_map = self._staging_params.pop(mid, {})
        if staging is None:
            raise RuntimeError(f"[expert_rotation] pre_sized: layer {mid} has no staging module")
        # Move the CPU staging params onto device for the format transform
        # (transient: R rows per layer, freed right after absorption).
        for name, param in staging.named_parameters():
            param.data = param.data.npu(non_blocking=False)
        # Same device-format transform the static side just went through
        # (NZ cast + transpose + scale layout), applied to the R-row twin.
        # Uses the UNWRAPPED original (see pre_create_layer) so this does not
        # re-enter split_layer through the fused_moe rotation wrap.
        self._orig_process_weights.pop(mid, None)(staging)

        components = {}
        from vllm_ascend.expert_rotation.weight_layout import (
            RotatingComponent,
            _rebuild_expert_rows,
            absorb_suffix_to_pinned,
            is_nz_tensor,
        )
        from vllm_ascend.expert_rotation.weight_layout import _ROLE_SPECS

        quant_type = runner.quant_type
        for (qt, role), spec in _ROLE_SPECS.items():
            if qt != quant_type:
                continue
            attr = next((a for a in spec.attrs if getattr(staging, a, None) is not None), None)
            if attr is None:
                continue
            t = getattr(staging, attr)
            if isinstance(t, torch.nn.Parameter):
                t = t.data
            if spec.materialize_view and not t.is_contiguous():
                from vllm_ascend.expert_rotation.weight_layout import _replace_layer_tensor

                _replace_layer_tensor(staging, attr, t.contiguous())
                t = getattr(staging, attr)
                if isinstance(t, torch.nn.Parameter):
                    t = t.data
            if t.shape[0] != num_rotating:
                raise RuntimeError(
                    f"[expert_rotation] pre_sized staging {attr} has {t.shape[0]} rows, "
                    f"expected {num_rotating}"
                )
            nz = is_nz_tensor(t)
            pinned = absorb_suffix_to_pinned(t, num_rotating)
            # merged mode: slot holds the combined S+R rows (static prefix +
            # rotating suffix); the staging tensor is only a shape/format
            # template for the rebuild, so S+R rows is fine even though the
            # staging module itself carries just R.
            slot_rows = runner.moe_config.num_local_experts if self.merged_gmm else num_rotating
            slots = [
                _rebuild_expert_rows(
                    t, slot_rows, nz=nz, transposed_view=spec.transposed_view, quant_type=quant_type
                )
                for _ in range(self.num_slots if first else 0)
            ]
            components[role] = RotatingComponent(
                role=role, attr=attr, pinned=pinned, slots=slots, nbytes=pinned.untyped_storage().nbytes()
            )
        num_local = runner.moe_config.num_local_experts
        state = LayerRotationState(
            moe_instance_id=mid,
            num_local_experts=num_local,
            num_static=num_local - num_rotating,
            num_rotating=num_rotating,
            components=components,
            pinned_bytes=sum(c.nbytes for c in components.values()),
            static_srcs=self._collect_static_srcs(runner.routed_experts, quant_type),
        )
        self.layers[mid] = state
        self._mark_layer(runner, state)
        self._pinned_bytes_total += state.pinned_bytes
        if first:
            for role, comp in components.items():
                self.slot_views[role] = comp.slots
            self._log_summary()
        # free the staging module (params + storage) eagerly
        del staging, param_map
        torch.npu.empty_cache()
        logger.info(
            "[expert_rotation] layer %d split (pre-sized): %d static + %d rotating, pinned %.2f GiB",
            mid,
            state.num_static,
            state.num_rotating,
            state.pinned_bytes / 1024**3,
        )

    def _split_layer_mega(self, runner, mid: int, num_rotating: int, first: bool) -> None:
        """MegaMoe variant of the split: the weights live as per-expert
        ``cann_mega_moe_*`` lists (the raw tensors were disposed by the quant
        method's process_weights_after_loading), so the split shrinks the
        layer's lists to the static S entries and absorbs the rotating R
        entries' bytes into pinned. The shared slot stays one combined R-row
        tensor per (role, slot) — the plain big-block H2D fill path — whose
        ``unbind(0)`` views are spliced after the static entries at run time
        (see :meth:`build_mega_weights`)."""
        layer = runner.routed_experts
        staging = None
        absorb_all = False
        if self.config.pre_sized and mid in self._staging_modules:
            staging = self._staging_modules.pop(mid)
            self._staging_params.pop(mid, None)
            # Move the CPU staging params onto device for the format transform
            # (transient: R rows per layer, freed right after absorption).
            for _, param in staging.named_parameters():
                param.data = param.data.npu(non_blocking=False)
            # Same UNWRAPPED transform the static side went through; under
            # MegaMoe this builds the staging module's cann_mega_moe_* lists
            # (R entries) and disposes its raw tensors, so the list absorb
            # below MUST run before any raw-tensor attr read on staging.
            self._orig_process_weights.pop(mid, None)(staging)
            absorb_all = True
            target = staging
        else:
            target = layer
        components = split_layer_weight_lists(
            target,
            runner.quant_type,
            num_rotating,
            num_slots=self.num_slots if first else 0,
            absorb_all=absorb_all,
            verify=self.config.verify_split,
        )
        num_local = runner.moe_config.num_local_experts
        # Snapshot the layer's (already shrunk, or pre-created S-sized) static
        # lists per role; the runtime hook splices the slot views after them.
        static_lists = {role: list(getattr(layer, comp.attr)) for role, comp in components.items()}
        state = LayerRotationState(
            moe_instance_id=mid,
            num_local_experts=num_local,
            num_static=num_local - num_rotating,
            num_rotating=num_rotating,
            components=components,
            pinned_bytes=sum(c.nbytes for c in components.values()),
            static_srcs={},  # static experts stay resident as list entries
            mega_static_lists=static_lists,
        )
        self.layers[mid] = state
        self._mark_layer(runner, state)
        self._pinned_bytes_total += state.pinned_bytes
        if first:
            for role, comp in components.items():
                self.slot_views[role] = comp.slots
            self._log_summary()
        # free the staging module (params + storage) eagerly
        del staging
        torch.npu.empty_cache()
        logger.info(
            "[expert_rotation] layer %d split (mega): %d static + %d rotating, pinned %.2f GiB",
            mid,
            state.num_static,
            state.num_rotating,
            state.pinned_bytes / 1024**3,
        )

    def _log_summary(self) -> None:
        slot_bytes = sum(
            sum(t.untyped_storage().nbytes() for t in views) for views in self.slot_views.values()
        )
        logger.info(
            "[expert_rotation] HBM rotation slots: %.2f GiB (%d slots x %d roles); "
            "static weights are the shrunk per-layer parameters",
            slot_bytes / 1024**3,
            self.num_slots,
            len(self.slot_views),
        )

    def log_final_summary(self) -> None:
        logger.info(
            "[expert_rotation] %d MoE layers split; pinned host total %.2f GiB; "
            "per-layer HBM routing-weight saving = %d/%d experts",
            len(self.layers),
            self._pinned_bytes_total / 1024**3,
            self.config.num_rotating_experts,
            next(iter(self.layers.values())).num_local_experts if self.layers else 0,
        )

    # ------------------------------------------------------------------
    # runtime
    # ------------------------------------------------------------------

    def make_runtime_ctx(self, moe_instance_id: int) -> RotationCtx | None:
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType

        state = self.layers.get(moe_instance_id)
        if state is None:
            # Layer never split (feature enabled after its load?) - must not
            # happen; the caller checks manager presence, we fail loudly here.
            raise RuntimeError(f"[expert_rotation] layer {moe_instance_id} was never split")
        if _EXTRA_CTX.capturing or torch.npu.is_current_stream_capturing():
            raise RuntimeError(
                "[expert_rotation] graph capture reached a rotation layer; expert rotation "
                "requires eager execution (cudagraph_mode=NONE / --enforce-eager)."
            )
        if _EXTRA_CTX.moe_comm_type == MoECommType.FUSED_MC2:
            if not _EXTRA_CTX.use_mega_moe:
                raise RuntimeError(
                    "[expert_rotation] the dispatch_ffn_combine variant of FUSED_MC2 "
                    "cannot receive rotating slot weights; use the MegaMoe variant "
                    "(additional_config enable_fused_mc2=2) or disable one of "
                    "enable_fused_mc2 / expert_rotation_config."
                )
            if not self.mega_mode:
                raise RuntimeError(
                    "[expert_rotation] FUSED_MC2+MegaMoe forward but the weights were "
                    "split with the tensor layout (not in mega list mode); the fused "
                    "path cannot consume them."
                )
            # mega fused path: the slot fill / prefetch chain below proceeds
            # exactly as on the split paths; the rotating experts enter the
            # fused op through build_mega_weights at the call site.
        elif self.mega_mode:
            raise RuntimeError(
                "[expert_rotation] weights were split in MegaMoe list mode but this "
                "forward selected a non-fused comm method; the GMM split path "
                "cannot consume list-split weights."
            )

        slot_idx = moe_instance_id % self.num_slots
        slot = self.slot_states[slot_idx]
        chain_ok = self._last_served is not None and moe_instance_id == self._last_served + 1
        if not chain_ok or slot.holder != moe_instance_id:
            # cold start (first layer of a batch), a chain break (draft restart)
            # or a stale slot: load this layer's block before its GMM.
            self._fill_slot(moe_instance_id, slot_idx)
        # keep up to num_slots-1 layers of lookahead in flight: with slot
        # ownership slot=s%num_slots, layer mid+k's block lives in slot
        # (mid+k)%num_slots — distinct from mid's own slot for k in
        # [1, num_slots-1], so a deeper pipeline never touches the slot the
        # current layer is reading. Each fill waits its slot_free event, so
        # device-side ordering stays correct for arbitrary depth.
        if self.config.prefetch_enabled:
            for k in range(1, self.num_slots):
                nxt = moe_instance_id + k
                if nxt not in self.layers:
                    break
                s = nxt % self.num_slots
                if self.slot_states[s].holder != nxt:
                    self._fill_slot(nxt, s)
        self._last_served = moe_instance_id
        return RotationCtx(
            manager=self,
            moe_instance_id=moe_instance_id,
            slot_idx=slot_idx,
            num_static=state.num_static,
            num_rotating=state.num_rotating,
        )

    def _fill_slot(self, moe_instance_id: int, slot_idx: int) -> None:
        """Load layer ``moe_instance_id``'s expert block into slot ``slot_idx``.

        Split mode: big-block H2D of the R rotating experts on the copy stream.
        Merged mode: the slot is the combined S+R buffer — the rotating suffix
        is H2D-filled from pinned on the copy stream WHILE the static prefix is
        D2D-gathered from the resident weights on the d2d stream (PCIe/SDMA and
        HBM/vector are distinct engines, so the two fills overlap; the D2D at
        ~670GB/s hides completely under the ~53GB/s H2D). copy_done fires only
        after both engines finish.

        Both modes wait for the slot's previous consumer (slot_free). Sync mode
        (prefetch disabled) enqueues on the compute stream itself, so the
        layer's own kernels are naturally ordered after the fill.
        """
        state = self.layers[moe_instance_id]
        slot = self.slot_states[slot_idx]
        # mega list mode: slots are plain R-row buffers (no S+R capacity, no
        # D2D static prefix) — always take the big-block H2D fill.
        merged_fill = self.merged_gmm and not self.mega_mode
        if self.copy_stream is not None:
            if not merged_fill:
                with torch.npu.stream(self.copy_stream):
                    self.copy_stream.wait_event(slot.slot_free)
                    for comp in state.components.values():
                        copy_storage_slice(
                            self.slot_views[comp.role][slot_idx],
                            comp.pinned.untyped_storage(),
                            non_blocking=True,
                        )
                    slot.copy_done.record(self.copy_stream)
            else:
                num_static, num_rotating = state.num_static, state.num_rotating
                with torch.npu.stream(self.copy_stream):
                    self.copy_stream.wait_event(slot.slot_free)
                    for comp in state.components.values():
                        slot_t = self.slot_views[comp.role][slot_idx]
                        per_e = bytes_per_expert(slot_t)
                        # rotating suffix: pinned H2D
                        copy_storage_slice(
                            slot_t,
                            comp.pinned.untyped_storage(),
                            non_blocking=True,
                            dst_offset=num_static * per_e,
                            nbytes=num_rotating * per_e,
                        )
                with torch.npu.stream(self.d2d_stream):
                    self.d2d_stream.wait_event(slot.slot_free)
                    for comp in state.components.values():
                        slot_t = self.slot_views[comp.role][slot_idx]
                        static_src = state.static_srcs[comp.role]
                        per_e = bytes_per_expert(slot_t)
                        # static prefix: resident D2D gather (src MUST be a
                        # storage slice: whole-storage NPU sources keep the
                        # internal-format tag and the copy raises)
                        copy_storage_slice(
                            slot_t,
                            static_src.untyped_storage(),
                            non_blocking=True,
                            dst_offset=0,
                            nbytes=num_static * per_e,
                        )
                    slot.d2d_done.record(self.d2d_stream)
                self.copy_stream.wait_event(slot.d2d_done)
                slot.copy_done.record(self.copy_stream)
        else:
            cur = torch.npu.current_stream()
            cur.wait_event(slot.slot_free)
            for comp in state.components.values():
                slot_t = self.slot_views[comp.role][slot_idx]
                if merged_fill:
                    num_static = state.num_static
                    per_e = bytes_per_expert(slot_t)
                    copy_storage_slice(
                        slot_t,
                        state.static_srcs[comp.role].untyped_storage(),
                        non_blocking=True,
                        dst_offset=0,
                        nbytes=num_static * per_e,
                    )
                    copy_storage_slice(
                        slot_t,
                        comp.pinned.untyped_storage(),
                        non_blocking=True,
                        dst_offset=num_static * per_e,
                        nbytes=state.num_rotating * per_e,
                    )
                else:
                    copy_storage_slice(
                        slot_t, comp.pinned.untyped_storage(), non_blocking=True
                    )
            slot.copy_done.record(cur)
        slot.holder = moe_instance_id

    def wait_slot_ready(self, ctx: RotationCtx) -> None:
        """Compute stream barrier before the rotating-segment GMM."""
        torch.npu.current_stream().wait_event(self.slot_states[ctx.slot_idx].copy_done)

    def mark_slot_consumed(self, ctx: RotationCtx) -> None:
        """Compute stream marker after the rotating-segment GMM is enqueued.

        MUST be called for every slot use so the next overwrite waits for the
        reader; see the event discipline in the module docstring.
        """
        slot = self.slot_states[ctx.slot_idx]
        slot.slot_free.record()
        ctx.consumed = True

    # ------------------------------------------------------------------
    # weight swapping for the rotating GMM call
    # ------------------------------------------------------------------

    def build_rot_weights(self, weights, slot_idx: int):
        """Swap the static MoEWeights payload for the slot views of ``slot_idx``.

        Preserves the list-vs-tensor convention each quant method used for the
        static side (e.g. W8A8 passes ``[tensor]`` single-element lists).
        """
        from dataclasses import replace

        def pick(role: str, cur: Any) -> Any:
            views = self.slot_views.get(role)
            if cur is None:
                return None
            if views is None:
                raise RuntimeError(f"[expert_rotation] no slot view for role {role}")
            slot_t = views[slot_idx]
            if isinstance(cur, list):
                return [slot_t] if len(cur) == 1 else [slot_t] * len(cur)
            return slot_t

        for bias_role in ("w1_bias", "w2_bias", "w1_offset", "w2_offset"):
            if getattr(weights, bias_role, None) is not None:
                raise RuntimeError(
                    f"[expert_rotation] {bias_role} is set; per-expert biases/offsets are "
                    "not supported by expert rotation yet."
                )

        return replace(
            weights,
            w1=pick("w1", weights.w1),
            w2=pick("w2", weights.w2),
            w1_scale=pick("w1_scale", weights.w1_scale),
            w2_scale=pick("w2_scale", weights.w2_scale),
        )

    def build_mega_weights(self, weights, slot_idx: int):
        """Splice the slot's rotating experts into a MegaMoe per-expert payload.

        The MegaMoe op consumes one per-expert tensor list covering ALL local
        experts in slot order; local ids [0,S) are the resident static entries
        (already in ``weights`` from get_fused_mc2_weights) and [S,S+R) are
        the slot tensor's ``unbind(0)`` views — the same view-over-combined-
        NZ-tensor form the op's W8A8Dynamic path ships. No remapping of
        topk_ids is needed: EP sharding is contiguous, so local slot order
        matches the global expert ids the op routes by.
        """
        from dataclasses import replace

        def merge(role: str, cur: Any) -> Any:
            if cur is None:
                return None
            views = self.slot_views.get(role)
            if views is None:
                raise RuntimeError(f"[expert_rotation] no slot view for role {role}")
            entries = list(views[slot_idx].unbind(0))
            if isinstance(cur, list):
                return list(cur) + entries
            return [cur] + entries

        for bias_role in ("w1_bias", "w2_bias", "w1_offset", "w2_offset"):
            if getattr(weights, bias_role, None) is not None:
                raise RuntimeError(
                    f"[expert_rotation] {bias_role} is set; per-expert biases/offsets are "
                    "not supported by expert rotation yet."
                )

        return replace(
            weights,
            w1=merge("w1", weights.w1),
            w2=merge("w2", weights.w2),
            w1_scale=merge("w1_scale", weights.w1_scale),
            w2_scale=merge("w2_scale", weights.w2_scale),
        )


# ----------------------------------------------------------------------
# process-wide singleton
# ----------------------------------------------------------------------
_EXPERT_ROTATION_MANAGER: ExpertRotationManager | None = None


def maybe_init_expert_rotation_manager() -> ExpertRotationManager | None:
    global _EXPERT_ROTATION_MANAGER
    if _EXPERT_ROTATION_MANAGER is not None:
        return _EXPERT_ROTATION_MANAGER
    from vllm_ascend.ascend_config import get_ascend_config

    config = get_ascend_config().expert_rotation_config
    if not config.enabled:
        return None
    _EXPERT_ROTATION_MANAGER = ExpertRotationManager(config)
    return _EXPERT_ROTATION_MANAGER


def get_expert_rotation_manager() -> ExpertRotationManager:
    assert _EXPERT_ROTATION_MANAGER is not None, "expert rotation manager not initialized"
    return _EXPERT_ROTATION_MANAGER


def has_expert_rotation_manager() -> bool:
    return _EXPERT_ROTATION_MANAGER is not None

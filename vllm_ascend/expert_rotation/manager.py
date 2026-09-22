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

from vllm_ascend.expert_rotation.config import ExpertRotationConfig
from vllm_ascend.expert_rotation.weight_layout import (
    RotatingComponent,
    copy_storage_slice,
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


class ExpertRotationManager:
    """Process-wide singleton owning slots, pinned blocks and the copy stream."""

    def __init__(self, config: ExpertRotationConfig):
        self.config = config
        self.num_slots = config.num_slots
        self.layers: dict[int, LayerRotationState] = {}
        # pre-sized mode: per-layer staging module + static->staging param map
        self._staging_modules: dict[int, torch.nn.Module] = {}
        self._staging_params: dict[int, dict[int, torch.nn.Parameter]] = {}
        self._orig_process_weights: dict[int, Any] = {}
        # slot views per MoEWeights role: role -> [slot0_view, slot1_view]
        self.slot_views: dict[str, list[torch.Tensor]] = {}
        self.slot_states: list[_SlotState] = [_SlotState() for _ in range(self.num_slots)]
        self.copy_stream = torch.npu.Stream() if config.prefetch_enabled else None
        self._last_served: int | None = None
        self._registered = 0
        self._pinned_bytes_total = 0
        # record the freeing events once so the very first wait cannot hang
        for s in self.slot_states:
            s.slot_free.record()
        logger.info(
            "[expert_rotation] manager created: num_slots=%d prefetch=%s",
            self.num_slots,
            config.prefetch_enabled,
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
        # once from the runner's expert map (tensor of shape [E_global]).
        emap = getattr(runner, "_expert_map", None)
        if emap is None:
            # ep_size == 1: identity mapping over the full expert set
            n_global = runner.moe_config.num_experts
            fwd = list(range(n_global))
        else:
            fwd = emap.cpu().tolist()
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
        if self.config.pre_sized and mid in self._staging_modules:
            self._split_pre_sized(runner, mid, num_rotating, first)
            return
        components = split_layer_weights(
            layer,
            runner.quant_type,
            num_rotating,
            num_slots=self.num_slots if first else 0,
            verify=self.config.verify_split,
        )
        num_local = runner.moe_config.num_local_experts
        state = LayerRotationState(
            moe_instance_id=mid,
            num_local_experts=num_local,
            num_static=num_local - num_rotating,
            num_rotating=num_rotating,
            components=components,
            pinned_bytes=sum(c.nbytes for c in components.values()),
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
            slots = [
                _rebuild_expert_rows(
                    t, num_rotating, nz=nz, transposed_view=spec.transposed_view, quant_type=quant_type
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
            raise RuntimeError(
                "[expert_rotation] FUSED_MC2 (fused dispatch_ffn_combine/mega_moe) cannot "
                "split the MLP stage; disable enable_fused_mc2 or expert_rotation_config."
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
        """Load layer ``moe_instance_id``'s rotating block into slot ``slot_idx``.

        Prefetch mode: big-block H2D on the copy stream, waiting for the slot's
        previous consumer to finish reading (slot_free). Sync mode: the copy is
        enqueued on the compute stream itself, so the layer's own kernels are
        naturally ordered after it.
        """
        state = self.layers[moe_instance_id]
        slot = self.slot_states[slot_idx]
        if self.copy_stream is not None:
            with torch.npu.stream(self.copy_stream):
                self.copy_stream.wait_event(slot.slot_free)
                for comp in state.components.values():
                    copy_storage_slice(
                        self.slot_views[comp.role][slot_idx], comp.pinned.untyped_storage(), non_blocking=True
                    )
                slot.copy_done.record(self.copy_stream)
        else:
            cur = torch.npu.current_stream()
            cur.wait_event(slot.slot_free)
            for comp in state.components.values():
                copy_storage_slice(
                    self.slot_views[comp.role][slot_idx], comp.pinned.untyped_storage(), non_blocking=True
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

"""Expert Offload Manager — manages CPU-side expert weights and NPU paging."""

import hashlib
import logging
import math
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch_npu
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import logger

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, get_moe_topk
from vllm_ascend.expert_offload.h2d_transfer import (
    CopyDirection,
    H2DCopyTask,
    HostPointerSource,
    TorchCopyH2DTransport,
    TorchSharedCPUWeightPool,
    create_h2d_transport,
)
from vllm_ascend.expert_offload.lrc_policy import LRCExpertCachePolicy
from vllm_ascend.ops.fused_moe.experts_selector import (
    commit_expert_substitutions,
    plan_expert_substitutions,
    substitute_experts,
)
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ


_SUBSCRIBED_COMPUTE_STREAMS = set()
_EXCLUSIVE_PREFILL_P2P_CHUNK_BYTES = 128 * 1024 * 1024
_FLOAT_CHECKSUM_NAN = 2_147_483_647
_FLOAT_CHECKSUM_POS_INF = 2_147_483_646
_FLOAT_CHECKSUM_NEG_INF = -2_147_483_646


def get_subscribed_compute_streams() -> set:
    return _SUBSCRIBED_COMPUTE_STREAMS


def _stable_int_checksum(values) -> int:
    """Return a deterministic checksum for small CPU-side integer tensors.

    Python's hash is process-randomized, so it cannot be compared across EP
    ranks. This helper is used only under ``moe_offload_debug`` and avoids an
    extra distributed collective in the decode path. Adding two preserves the
    distinction between the ``-1`` log2phy sentinel and physical slot zero.
    """
    flat = values.reshape(-1).tolist() if hasattr(values, "reshape") else values
    return sum((index + 1) * (int(value) + 2)
               for index, value in enumerate(flat))


def _stable_float_checksum(values, scale: int = 1_000_000) -> int:
    """Deterministic fixed-point checksum for cross-rank debug comparison.

    Router scores are diagnostic inputs and may already contain NaN/Inf when
    the model produces a non-finite activation. Debug logging must report that
    condition instead of raising while converting it to an integer.
    """
    flat = values.reshape(-1).tolist() if hasattr(values, "reshape") else values
    checksum = 0
    for index, value in enumerate(flat):
        number = float(value)
        if math.isnan(number):
            quantized = _FLOAT_CHECKSUM_NAN
        elif math.isinf(number):
            quantized = (_FLOAT_CHECKSUM_POS_INF if number > 0 else
                         _FLOAT_CHECKSUM_NEG_INF)
        else:
            scaled = number * scale
            if math.isinf(scaled):
                quantized = (_FLOAT_CHECKSUM_POS_INF if scaled > 0 else
                             _FLOAT_CHECKSUM_NEG_INF)
            else:
                quantized = round(scaled)
        checksum += (index + 1) * quantized
    return checksum


def _expert_weight(layer, name: str):
    """Resolve an expert weight tensor, packed-aware.

    ascend w4a8_mxfp4 with ``use_weight_packed=True`` (e.g. Kimi-K3 routed via
    compressed-tensors) stores weights as ``w13_weight_packed`` / ``w2_weight_packed``;
    otherwise plain ``w13_weight`` / ``w2_weight``. Scales (``*_scale``) are never
    packed. Return whichever attribute exists.
    """
    packed = getattr(layer, name + "_packed", None)
    if packed is not None:
        return packed
    return getattr(layer, name)


@dataclass(frozen=True)
class _PrefillPeerComponent:
    """One deterministic component in a peer communication buffer.

    Internal-format weights use raw storage. Quantization metadata retains
    its typed tensor views so local pack/unpack copies preserve non-contiguous
    logical layouts while HCCL still sees only the enclosing uint8 buffer.
    """

    name: str
    source: object
    destination: object
    nbytes: int
    dtype: torch.dtype | None = None
    shape: tuple[int, ...] | None = None
    element_size: int = 1

    @property
    def uses_typed_copy(self) -> bool:
        return self.dtype is not None


class ExpertOffloadManager:
    """Singleton manager for expert weight offloading.

    In legacy mode, stores all expert weights on CPU and pages the needed
    experts to NPU.  In ``exclusive_dynamic`` mode, CPU and NPU instead own
    disjoint canonical expert sets and misses are served by two-way swaps.
    """

    _instance: "ExpertOffloadManager | None" = None

    # Parallel weight-load pool. The strided transpose-copy in load_w13/
    # load_w2 is single-threaded (~0.2 GB/s into pinned memory); fanning the
    # ~99k shard copies out over this many workers hits ~2-4 GB/s.
    _LOAD_POOL_WORKERS = 32
    # Bound on in-flight futures before a partial drain (releases owned clones
    # early so transient memory stays small). >> workers, so no starvation.
    # 128 (= 4x workers): for big MoE (Kimi-K3 1.5T) the old 2048 piled ~0.8T
    # of owned clones on top of the 1.5T pinned buffer → host peak ~2.3T (near
    # the 2.9T limit). 128 bounds in-flight clones to ~50G with no throughput
    # loss (pool still runs 32-wide; only the drain point is denser).
    _LOAD_POOL_DRAIN_EVERY = 128
    # Keep the worker barrier dense for host-memory safety, but report loading
    # progress much less frequently. A final aggregate is always logged by
    # _finalize_offload, so small models do not need intermediate progress.
    _LOAD_PROGRESS_LOG_EVERY = 4096
    _DEBUG_EXPERT_SAMPLE_LIMIT = 16
    _SHARED_CPU_PSS_RATIO_TOLERANCE = 0.05
    _SHARED_CPU_MAPPING_SIZE_TOLERANCE = 0.01

    @classmethod
    def get_instance(cls) -> "ExpertOffloadManager":
        assert cls._instance is not None, "ExpertOffloadManager not initialized"
        return cls._instance

    def __init__(self, vllm_config: VllmConfig):
        from vllm_ascend.ascend_config import get_ascend_config

        self.offload_config = get_ascend_config().expert_offload_config
        # The minimum capacity is the conservative global dispatch threshold;
        # actual weight and placement sizes are resolved per MoE layer.
        self.num_device_experts = min(
            self.offload_config.num_device_experts_list)
        # Match the FusedMoE layer, which is built from hf_text_config. A
        # multimodal top-level config may retain a different, stale TopK.
        self.topk = get_moe_topk(vllm_config)
        assert self.topk, ("offload: cannot find num_experts_per_tok/num_experts_per_token "
                           "on hf_text_config")
        self.topk = int(self.topk)
        self.offload_threshold = self.num_device_experts // self.topk

        # Multi-card EP offload (stages 1-2). ep_rank/ep_size are resolved
        # lazily on first read because the EP process group is not initialized
        # yet at manager construction time (model_runner.__init__ runs before
        # init_distributed_environment completes).
        self.enable_multi_card = self.offload_config.enable_multi_card
        self._ep_size = 1
        self._ep_rank = 0
        self._ep_info_resolved = False
        # Multi-card decode resident cache: per layer_idx, {slot: expert_id} of
        # the experts currently loaded in THIS rank's device slots. Used to turn
        # the per-step full H2D into skip-on-hit (only load misses) and to log
        # hit/miss. Keyed by slot (the planner assigns expert->slot via log2phy,
        # so a hit = same expert already in the same slot).
        self._mc_resident = {}
        # Two-timescale LRU: per-step local freq tracking + every-N-step gloo
        # all_reduce -> global hotness. Stable-slot placement uses prev_log2phy
        # (keep experts on same rank in their slot) + hotness (order new experts).
        self._mc_prev_log2phy = {}      # layer_idx -> prev step's log2phy (CPU)
        # LRC hotness policy (same one single-card uses: recent freq + EMA +
        # age), fed the GLOBAL active set each step. Built lazily in
        # _gather_global_counts_and_hotness. Replaces the old crude local-freq
        # + 32-step all_reduce hotness, which was stale and had no EMA/age.
        self._mc_lrc = None

        # Per-layer cap on experts actually H2D-loaded by _do_prefetch: only
        # the top-N highest-confidence predicted experts are loaded, the rest
        # are left to update_weights()'s reactive fallback. Clamped to
        # [1, topk]; >topk has no extra effect since the router selects at
        # most topk experts per token.
        self.expert_prefetch_num = self.offload_config.expert_prefetch_num
        self.prefetch_topk = max(1, min(self.topk, self.expert_prefetch_num))

        # CPU weight buffers (post-transpose format, matching device after
        # process_weights_after_loading):
        #   w13 per expert: [hidden_size, w13_up_dim]
        #   w2 per expert:  [intermediate_size_per_partition, hidden_size]
        self.w13_weights_cpu: list[list[torch.Tensor]] = []
        self.w2_weights_cpu: list[list[torch.Tensor]] = []

        # Registered AscendFusedMoE layers, indexed by moe_instance_id order
        self.moe_layers: list = []

        # CPU buffers for quantized model scale/offset parameters.
        # Keyed by attr_name (e.g. "w13_weight_scale", "w2_weight_offset").
        # Each value is a list of layers, each layer is a list of expert tensors.
        self.scale_cpu_buffers: dict[str, list[list[torch.Tensor]]] = {}
        self.offset_cpu_buffers: dict[str, list[list[torch.Tensor]]] = {}
        self.scale_bias_cpu_buffers: dict[str, list[list[torch.Tensor]]] = {}

        # Dynamic exclusive ownership. Every expert has exactly one canonical
        # location per layer: a compact CPU slot or an NPU physical slot.
        # Single-card uses private pinned CPU tensors. Multi-card either uses
        # one same-node mmap plus a global physical-slot namespace, or private
        # rank-local CPU/NPU partitions whose union is that rank's EP shard.
        # Runtime swaps update all four inverse maps only after D2H + H2D have
        # synchronized successfully.
        self._cpu_slot_to_eid: list[list[int]] = []
        self._eid_to_cpu_slot: list[list[int]] = []
        self._npu_slot_to_eid: list[list[int]] = []
        self._eid_to_npu_slot: list[list[int]] = []
        self._exclusive_layer_locks: list[threading.RLock] = []
        # Multi-card exclusive mode maps one compact shared CPU pool on every
        # rank. Checkpoint writes are divided by compact CPU slot so each
        # canonical CPU expert is initialized exactly once.
        self._exclusive_shared_local_cpu_slots: list[list[int]] = []
        self._exclusive_shared_cpu_slot_to_local: list[dict[int, int]] = []
        # Global shared-exclusive mode uses the same initialization discipline
        # as rank-local exclusive mode: initial NPU owners are checkpointed into
        # a temporary rank-local CPU suffix, pass through the same weight/quant
        # post-processing as compact shared-CPU owners, and are then copied to
        # this rank's NPU slots.  The suffix is released before runtime memory
        # accounting, preserving one global canonical CPU+NPU model.
        self._exclusive_shared_bootstrap_layers: set[int] = set()
        # Rank-local exclusive mode must put the initial NPU owners through the
        # same CPU post-processing path as later H2D misses.  During checkpoint
        # loading each layer therefore appends one temporary CPU tensor per
        # initial NPU slot after its compact canonical CPU tensors.  Finalize
        # materializes those processed bytes on the local NPU and immediately
        # drops the temporary suffix, restoring CPU + NPU == one local shard.
        self._exclusive_sharded_bootstrap_layers: set[int] = set()
        # Cross-rank prefill transfers cannot pass internal-format expert
        # slices through torch.distributed because those views commonly have
        # a non-zero storage_offset and ProcessGroupHCCL owns a separate
        # communication stream. Serialize buffer reuse and use a dedicated
        # raw-pointer PyHCCL communicator on the manager's load stream.
        self._exclusive_prefill_comm_lock = threading.Lock()
        self._exclusive_prefill_comm_buffers: dict[
            tuple[str, int], torch.Tensor] = {}
        self._exclusive_prefill_pyhccl = None

        # Temporary per-expert storage for w13 scale/offset shard assembly.
        # Key: (layer_moe_idx, expert_id, attr_name), value: first shard.
        # Scale/offset arrive as w1 + w3 shards; we stash one until the
        # other arrives, then assemble and copy into scale_cpu_buffers.
        self._scale_shard_temp: dict[tuple[int, int, str], torch.Tensor] = {}

        self.num_device_layers = self.offload_config.num_device_layers
        self.num_total_experts = None  # set in init_layer_cpu_buffers
        self.cache_policy: LRCExpertCachePolicy | None = None
        self.cache_requests: list[int] = []
        self.cache_hits: list[int] = []
        self.cache_misses: list[int] = []
        self.cache_calls: list[int] = []
        self.last_hit_experts: list[list[int]] = []
        self.last_miss_experts: list[list[int]] = []
        # Master debug switch for expert-offload diagnostics — UPDATE-W cache
        # trace, per-prefill-load logs, prefetch/update slot shortfalls.
        # Flipping it on surfaces them at info level (no need for global
        # VLLM_LOGGING_LEVEL=DEBUG).
        self._debug = self.offload_config.moe_offload_debug
        # Graph/collective diagnostics. Host callbacks may execute on report
        # threads, so keep the counters under a small CPU-only lock. These
        # fields are touched only when moe_offload_debug is enabled and never
        # read device tensors or introduce another distributed collective.
        self._mc_debug_lock = threading.Lock()
        self._mc_debug_schedule_seq = 0
        self._mc_debug_callback_seq = 0
        self._mc_debug_collective_seq = 0
        self._mc_debug_active_callbacks = 0
        self._mc_debug_layer_calls: dict[tuple[int, bool], int] = {}

        # Diagnostic: wall time of the parallel weight-load phase (safetensors
        # → pinned CPU buffers). Logged in _finalize_offload.
        self._weight_load_secs: float = 0.0
        self._weight_load_calls: int = 0

        # Deferred weight-load pool. load_w13/load_w2/_load_scale_shard clone
        # loaded_weight synchronously (while the safetensors mmap is still
        # mapped) and submit the strided transpose-copy to this pool. The
        # deferred copy reads the owned clone, so it stays correct after the
        # safetensors mmap is unmapped (which happens before _finalize_offload).
        # drain_load_pool() is called from _finalize_offload before the buffers
        # are read by process_weights_after_loading().
        self._load_pool: ThreadPoolExecutor | None = None
        self._load_futures: list = []
        self._load_phase_start: float = 0.0
        self._saved_num_threads: int | None = None

        ExpertOffloadManager._instance = self

        self.load_stream = torch_npu.npu.Stream()
        # MemFabric SHARED initialization is collective and needs the EP group,
        # so defer it until the first layer allocates Host expert buffers.
        self.h2d_transport = None
        if not (self.offload_config.h2d_backend == "memfabric"
                and self.enable_multi_card):
            self.h2d_transport = self._create_h2d_transport()
        self._shared_h2d_sources = {}
        self._shared_h2d_sources_ready = False
        self._torch_shared_cpu_pool = None
        self._torch_shared_cpu_buffers: dict[
            tuple[int, str], torch.Tensor] = {}
        self._torch_shared_cpu_sources_ready = False
        self._torch_shared_cpu_ready_layers: set[int] = set()
        logger.info(
            "[EXPERT-OFFLOAD-H2D] backend=%s mode=%s "
            "memfabric_pool_size_gib=%d",
            self.offload_config.h2d_backend,
            ("torch-shared-staging"
             if self.torch_shared_cpu_weights_enabled else
             ("shared" if self.enable_multi_card else "local")
             if self.offload_config.h2d_backend == "memfabric" else "copy"),
            self.offload_config.memfabric_pool_size_gib,
        )

        self._init_prefill_pool_state()
        self._is_prefetch: bool = False
        self._init_prefetch_state()

    def _init_prefill_pool_state(self) -> None:
        """Prefill-pool attribute init (ndl layers × all experts on NPU)."""
        # Prefill pool: ndl layers × all experts on NPU, shared round-robin
        self._prefill_w13: list[torch.Tensor] = []
        self._prefill_w2: list[torch.Tensor] = []
        self._prefill_w13_scale: list[torch.Tensor] = []       # W8A8 / W4A8_DYNAMIC
        self._prefill_w13_scale_fp32: list[torch.Tensor] = []   # W8A8
        self._prefill_w13_offset: list[torch.Tensor] = []       # W8A8
        self._prefill_w2_scale: list[torch.Tensor] = []         # W8A8 / W4A8_DYNAMIC
        self._prefill_w2_offset: list[torch.Tensor] = []        # W8A8
        # W4A8_DYNAMIC scale_bias (float32), per-channel new_quant_version only.
        # Allocated lazily in create_prefill_pool when the layer has
        # w13_scale_bias / w2_scale_bias parameters.
        self._prefill_w13_scale_bias: list[torch.Tensor] = []
        self._prefill_w2_scale_bias: list[torch.Tensor] = []
        self._prefill_log2phy: torch.Tensor = None              # identity [0..127]
        # Multi-card All2All temporarily expands the dispatcher from the
        # decode cache size to the full EP shard. Build its CPU metadata and
        # NPU mapping once, before ACL graph capture; apply() only swaps refs.
        self._prefill_local_expert_indices: list[int] | None = None
        self._prefill_expert_ids_per_ep_rank: torch.Tensor | None = None
        self._prefill_initialized: bool = False
        self._skip_prefill: bool = False  # set during profile runs

    def _init_prefetch_state(self) -> None:
        """Next-layer expert-prefetch infrastructure init."""
        # Next-layer expert prefetch infrastructure
        self._prefetch_stream = torch_npu.npu.Stream()
        # NPU copy of gate weights for graph-capturable on-device prediction
        # (predict_next_layer_experts_npu). Kept in fp32.
        self._gate_weights_npu: list[torch.Tensor | None] = []

        # Prefetch state: _prefetch_state_lock guards _prefetch_layer_npu_event,
        # which is shared by the forward thread and the graph host callback.
        self._prefetch_state_lock = threading.Lock()
        self._prefetch_layer_npu_event: dict[int, torch_npu.npu.Event] = {}

        # Pinned CPU staging buffer for graph-mode prefetch: trigger_next_
        # layer_prefetch stages the next layer's log2phy here with
        # non_blocking D2H around the host callback, mirroring update_weights
        # (blocking .cpu() on a live graph tensor would deadlock on replay).
        # Allocated lazily in _finalize_offload (num_total_experts is only
        # known after MoE layers register).
        self._prefetch_log2phy_h: torch.Tensor | None = None
        self._prefetch_log2phy_np = None

    def _resolve_ep_info(self) -> None:
        """Lazily resolve ep_rank/ep_size from the EP group on first access.

        No-op (stays ep_size=1, ep_rank=0) when ``enable_multi_card`` is False,
        so the single-card path is unchanged.
        """
        if self._ep_info_resolved:
            return
        if self.enable_multi_card:
            from vllm.distributed.parallel_state import get_ep_group
            ep_group = get_ep_group()
            self._ep_size = ep_group.world_size
            self._ep_rank = ep_group.rank_in_group
        self._ep_info_resolved = True

    @property
    def ep_size(self) -> int:
        self._resolve_ep_info()
        return self._ep_size

    @property
    def ep_rank(self) -> int:
        self._resolve_ep_info()
        return self._ep_rank

    def _create_h2d_transport(self):
        enable_shared = (
            self.offload_config.h2d_backend == "memfabric"
            and self.enable_multi_card)
        return create_h2d_transport(
            self.offload_config.h2d_backend,
            device_id=torch_npu.npu.current_device(),
            memfabric_pool_size_gib=(
                self.offload_config.memfabric_pool_size_gib),
            memfabric_log_level=self.offload_config.memfabric_log_level,
            enable_multi_card=enable_shared,
            enable_torch_shared_cpu=self.torch_shared_cpu_weights_enabled,
            world_size=self.ep_size if enable_shared else 1,
            rank_id=self.ep_rank if enable_shared else 0,
        )

    # ------------------------------------------------------------------ #
    #  Lifecycle: called during model init and after weight loading       #
    # ------------------------------------------------------------------ #

    def num_device_experts_for_layer(self, layer_idx: int) -> int:
        """Per-layer device-expert buffer size (delegates to offload config).

        The config is the single source of truth: a scalar broadcasts, a list
        indexes by MoE-layer registration order.
        """
        return self.offload_config.num_device_experts_for_layer(layer_idx)

    @property
    def exclusive_dynamic_enabled(self) -> bool:
        """Whether CPU and NPU use disjoint, dynamically swapped ownership."""
        return (self.offload_config.storage_partition_mode
                == "exclusive_dynamic")

    @property
    def torch_shared_cpu_weights_enabled(self) -> bool:
        """Whether canonical CPU expert data uses one same-node Torch mmap."""
        return bool(self.offload_config.shared_cpu_weights)

    @property
    def exclusive_shared_cpu_enabled(self) -> bool:
        """Whether exclusive ownership spans a same-node EP group."""
        return (self.exclusive_dynamic_enabled
                and self.torch_shared_cpu_weights_enabled
                and getattr(self, "enable_multi_card",
                            self.offload_config.enable_multi_card))

    @property
    def exclusive_sharded_cpu_enabled(self) -> bool:
        """Whether exclusive ownership is private and EP-shard local."""
        return (self.exclusive_dynamic_enabled
                and self.offload_config.shard_per_rank
                and not self.torch_shared_cpu_weights_enabled
                and getattr(self, "enable_multi_card",
                            self.offload_config.enable_multi_card))

    def _ensure_torch_shared_cpu_pool(self) -> TorchSharedCPUWeightPool:
        if self._torch_shared_cpu_pool is None:
            from vllm.distributed.parallel_state import get_ep_group
            self._torch_shared_cpu_pool = TorchSharedCPUWeightPool(
                self.offload_config.shared_cpu_weights_dir,
                world_size=self.ep_size,
                rank_id=self.ep_rank,
                cpu_group=get_ep_group().cpu_group,
            )
            logger.info(
                "[EXPERT-OFFLOAD-H2D] initialized same-node Torch shared "
                "CPU pool: rank=%d/%d dir=%s",
                self.ep_rank, self.ep_size,
                self.offload_config.shared_cpu_weights_dir)
        return self._torch_shared_cpu_pool

    def _allocate_torch_shared_expert_buffer(
        self,
        layer_idx: int,
        name: str,
        per_expert_shape,
        dtype,
    ) -> list[torch.Tensor]:
        """Allocate one canonical mmap and return this rank's writer views."""
        if self.exclusive_shared_cpu_enabled:
            global_experts = len(self._cpu_slot_to_eid[layer_idx])
            local_slots = self._exclusive_shared_local_cpu_slots[layer_idx]
        else:
            global_experts = self.num_total_experts
            local_slots = range(self._shard_base,
                                self._shard_base + self._shard_size)
        global_buffer = self._ensure_torch_shared_cpu_pool().allocate(
            f"L{layer_idx}-{name}",
            (global_experts,) + tuple(per_expert_shape),
            dtype,
        )
        self._torch_shared_cpu_buffers[(layer_idx, name)] = global_buffer
        return [global_buffer[slot] for slot in local_slots]

    def _replace_torch_shared_expert_buffer(
        self,
        layer_idx: int,
        name: str,
        local_values: list[torch.Tensor],
        template: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Replace a post-processed quant buffer with another shared mmap."""
        shared_count = (
            len(self._exclusive_shared_local_cpu_slots[layer_idx])
            if self.exclusive_shared_cpu_enabled else self._shard_size)
        bootstrap_count = 0
        if (self.exclusive_shared_cpu_enabled
                and layer_idx in getattr(
                    self, "_exclusive_shared_bootstrap_layers", set())):
            bootstrap_count = self.offload_config.num_device_experts_for_rank(
                layer_idx, self.ep_size)
        expected = shared_count + bootstrap_count
        if len(local_values) != expected:
            raise RuntimeError(
                "Torch shared CPU replacement must cover the local shard: "
                f"layer={layer_idx}, name={name}, values={len(local_values)}, "
                f"shared={shared_count}, bootstrap={bootstrap_count}")
        if template is None:
            if not local_values:
                raise RuntimeError(
                    "Torch shared CPU replacement needs a template when "
                    f"the local writer shard is empty: layer={layer_idx}, "
                    f"name={name}")
            template = local_values[0]
        if any(value.shape != template.shape or value.dtype != template.dtype
               for value in local_values):
            raise RuntimeError(
                "Torch shared CPU replacement tensors must have one shape "
                f"and dtype: layer={layer_idx}, name={name}")
        destinations = self._allocate_torch_shared_expert_buffer(
            layer_idx, name, template.shape, template.dtype)
        for destination, value in zip(
                destinations, local_values[:shared_count]):
            destination.copy_(value)
        for value in local_values[shared_count:]:
            destination = self._allocate_expert_host_tensor(
                value.shape, value.dtype)
            destination.copy_(value)
            destinations.append(destination)
        return destinations

    def _init_exclusive_ownership(
        self,
        layer_moe_idx: int,
        ntotal: int,
        ndev: int,
        npu_eids: list[int] | None = None,
        owned_eids: list[int] | None = None,
    ) -> None:
        """Create inverse CPU/NPU ownership maps for one MoE layer."""
        if not 0 < ndev < ntotal:
            raise ValueError(
                "exclusive_dynamic requires 0 < num_device_experts < "
                f"global_num_experts; layer={layer_moe_idx}, "
                f"num_device_experts={ndev}, global_num_experts={ntotal}")

        npu_slot_to_eid = (
            list(range(ndev)) if npu_eids is None else list(npu_eids))
        if (len(npu_slot_to_eid) != ndev
                or len(set(npu_slot_to_eid)) != ndev
                or any(eid < 0 or eid >= ntotal for eid in npu_slot_to_eid)):
            raise ValueError(
                "Initial exclusive NPU experts must be unique, in range, and "
                f"match device capacity; layer={layer_moe_idx}, "
                f"ndev={ndev}, experts={npu_slot_to_eid}")
        canonical_eids = (
            list(range(ntotal)) if owned_eids is None else list(owned_eids))
        if (len(set(canonical_eids)) != len(canonical_eids)
                or any(eid < 0 or eid >= ntotal for eid in canonical_eids)):
            raise ValueError(
                "Exclusive canonical experts must be unique and in range: "
                f"layer={layer_moe_idx}, experts={canonical_eids}")
        npu_set = set(npu_slot_to_eid)
        if not npu_set.issubset(canonical_eids):
            raise ValueError(
                "Initial exclusive NPU experts must belong to this rank's "
                f"canonical set: layer={layer_moe_idx}, npu={npu_slot_to_eid}, "
                f"canonical={canonical_eids}")
        cpu_slot_to_eid = [
            eid for eid in canonical_eids if eid not in npu_set]
        eid_to_npu_slot = [-1] * ntotal
        eid_to_cpu_slot = [-1] * ntotal
        for slot, eid in enumerate(npu_slot_to_eid):
            eid_to_npu_slot[eid] = slot
        for slot, eid in enumerate(cpu_slot_to_eid):
            eid_to_cpu_slot[eid] = slot

        if layer_moe_idx != len(self._cpu_slot_to_eid):
            raise RuntimeError(
                "MoE layers must register in contiguous order for "
                f"exclusive ownership; got layer={layer_moe_idx}, "
                f"registered={len(self._cpu_slot_to_eid)}")
        self._cpu_slot_to_eid.append(cpu_slot_to_eid)
        self._eid_to_cpu_slot.append(eid_to_cpu_slot)
        self._npu_slot_to_eid.append(npu_slot_to_eid)
        self._eid_to_npu_slot.append(eid_to_npu_slot)
        self._exclusive_layer_locks.append(threading.RLock())
        self._validate_exclusive_ownership(layer_moe_idx)

    def _validate_exclusive_ownership(self, layer_idx: int) -> None:
        """Validate the CPU/NPU partition and both inverse maps."""
        cpu_eids = self._cpu_slot_to_eid[layer_idx]
        npu_eids = self._npu_slot_to_eid[layer_idx]
        ntotal = len(self._eid_to_cpu_slot[layer_idx])
        if self.exclusive_sharded_cpu_enabled:
            start = self._shard_base
            expected_eids = set(range(start, start + self._shard_size))
        else:
            expected_eids = set(range(ntotal))
        if (len(cpu_eids) + len(npu_eids) != len(expected_eids)
                or set(cpu_eids).intersection(npu_eids)
                or set(cpu_eids).union(npu_eids) != expected_eids):
            raise RuntimeError(
                f"Invalid exclusive expert partition for layer {layer_idx}")
        for slot, eid in enumerate(cpu_eids):
            if self._eid_to_cpu_slot[layer_idx][eid] != slot:
                raise RuntimeError(
                    f"Invalid CPU ownership inverse for layer={layer_idx}, "
                    f"expert={eid}, slot={slot}")
            if self._eid_to_npu_slot[layer_idx][eid] != -1:
                raise RuntimeError(
                    f"Expert {eid} is owned by both CPU and NPU in layer "
                    f"{layer_idx}")
        for slot, eid in enumerate(npu_eids):
            if self._eid_to_npu_slot[layer_idx][eid] != slot:
                raise RuntimeError(
                    f"Invalid NPU ownership inverse for layer={layer_idx}, "
                    f"expert={eid}, slot={slot}")
            if self._eid_to_cpu_slot[layer_idx][eid] != -1:
                raise RuntimeError(
                    f"Expert {eid} is owned by both NPU and CPU in layer "
                    f"{layer_idx}")
        for eid in set(range(ntotal)) - expected_eids:
            if (self._eid_to_cpu_slot[layer_idx][eid] != -1
                    or self._eid_to_npu_slot[layer_idx][eid] != -1):
                raise RuntimeError(
                    "Remote expert unexpectedly has local exclusive ownership: "
                    f"layer={layer_idx}, expert={eid}")

    def init_layer_cpu_buffers(self, layer, layer_moe_idx: int):
        """Allocate CPU weight + scale/offset buffers for one MoE layer.

        Called from AscendFusedMoE.__init__ after device tensors are set up,
        so CPU buffers exist before the safetensors weight loader runs.
        """
        ntotal = layer.global_num_experts
        if self.num_total_experts is None:
            self.num_total_experts = ntotal
        assert ntotal == self.num_total_experts, \
            f"MoE layers must have same expert count: {ntotal} vs {self.num_total_experts}"

        _w13 = _expert_weight(layer, "w13_weight")
        _w2 = _expert_weight(layer, "w2_weight")
        params_dtype = _w13.dtype
        w13_shape = (_w13.shape[2], _w13.shape[1])
        w2_shape = (_w2.shape[2], _w2.shape[1])

        ndev = self.num_device_experts_for_layer(layer_moe_idx)
        if self.exclusive_dynamic_enabled:
            multi_card_exclusive = (
                self.exclusive_shared_cpu_enabled
                or self.exclusive_sharded_cpu_enabled)
            device_slots = (
                self.offload_config.num_device_experts_for_rank(
                    layer_moe_idx, self.ep_size)
                if multi_card_exclusive else ndev)
            if (_w13.shape[0] != device_slots
                    or _w2.shape[0] != device_slots):
                raise ValueError(
                    "exclusive_dynamic device expert buffer does not match "
                    f"num_device_experts: layer={layer_moe_idx}, "
                    f"configured_global={ndev}, "
                    f"configured_per_rank={device_slots}, "
                    f"w13_slots={_w13.shape[0]}, "
                    f"w2_slots={_w2.shape[0]}")
            if self.exclusive_sharded_cpu_enabled:
                if ntotal % self.ep_size != 0:
                    raise ValueError(
                        "Rank-local exclusive CPU weights require "
                        "global_num_experts divisible by EP size: "
                        f"experts={ntotal}, ep_size={self.ep_size}")
                self._shard_size = ntotal // self.ep_size
                self._shard_base = self.ep_rank * self._shard_size
                initial_npu_eids = (
                    self.offload_config.initial_device_experts_for_rank(
                        layer_moe_idx, ntotal, self.ep_size, self.ep_rank))
                owned_eids = list(range(
                    self._shard_base, self._shard_base + self._shard_size))
                self._init_exclusive_ownership(
                    layer_moe_idx, ntotal, device_slots, initial_npu_eids,
                    owned_eids=owned_eids)
            else:
                initial_npu_eids = (
                    self.offload_config.initial_device_experts_for_layer(
                        layer_moe_idx, ntotal))
                self._init_exclusive_ownership(
                    layer_moe_idx, ntotal, ndev, initial_npu_eids)
            cpu_experts = len(self._cpu_slot_to_eid[layer_moe_idx])
            if self.exclusive_shared_cpu_enabled:
                if ntotal % self.ep_size != 0:
                    raise ValueError(
                        "Multi-card exclusive Torch shared CPU weights "
                        "require global_num_experts divisible by EP size: "
                        f"experts={ntotal}, ep_size={self.ep_size}")
                self._shard_size = ntotal // self.ep_size
                self._shard_base = self.ep_rank * self._shard_size
                start = cpu_experts * self.ep_rank // self.ep_size
                end = cpu_experts * (self.ep_rank + 1) // self.ep_size
                local_slots = list(range(start, end))
                self._exclusive_shared_local_cpu_slots.append(local_slots)
                self._exclusive_shared_cpu_slot_to_local.append({
                    slot: local for local, slot in enumerate(local_slots)
                })
                w13_list = self._allocate_torch_shared_expert_buffer(
                    layer_moe_idx, "w13", w13_shape, params_dtype)
                w2_list = self._allocate_torch_shared_expert_buffer(
                    layer_moe_idx, "w2", w2_shape, params_dtype)
                # Append only this rank's initial NPU owners. These private
                # tensors exist during checkpoint loading/finalization and are
                # removed after their uniformly post-processed bytes are copied
                # to the corresponding local device slots.
                if not hasattr(self, "_exclusive_shared_bootstrap_layers"):
                    self._exclusive_shared_bootstrap_layers = set()
                self._exclusive_shared_bootstrap_layers.add(layer_moe_idx)
                w13_list.extend(
                    self._allocate_expert_host_tensor(w13_shape, params_dtype)
                    for _ in range(device_slots))
                w2_list.extend(
                    self._allocate_expert_host_tensor(w2_shape, params_dtype)
                    for _ in range(device_slots))
            else:
                # The suffix is initialization-only.  It captures the experts
                # whose canonical owner is already NPU so they receive exactly
                # the same transpose/NZ/scale processing as CPU-owned experts.
                # _materialize_exclusive_sharded_bootstrap removes it before
                # runtime and before memory accounting.
                checkpoint_experts = cpu_experts
                if self.exclusive_sharded_cpu_enabled:
                    checkpoint_experts += device_slots
                    if not hasattr(self, "_exclusive_sharded_bootstrap_layers"):
                        self._exclusive_sharded_bootstrap_layers = set()
                    self._exclusive_sharded_bootstrap_layers.add(layer_moe_idx)
                w13_list = [
                    self._allocate_expert_host_tensor(w13_shape, params_dtype)
                    for _ in range(checkpoint_experts)
                ]
                w2_list = [
                    self._allocate_expert_host_tensor(w2_shape, params_dtype)
                    for _ in range(checkpoint_experts)
                ]
            self.w13_weights_cpu.append(w13_list)
            self.w2_weights_cpu.append(w2_list)
        elif self.torch_shared_cpu_weights_enabled:
            # Checkpoint ownership stays sharded: rank r writes only
            # [r*shard:(r+1)*shard). All ranks map the same global backing
            # tensors, so decode can read any expert after finalization.
            if ntotal % self.ep_size != 0:
                raise ValueError(
                    "Torch shared CPU weights require global_num_experts "
                    "to be divisible by EP size: "
                    f"experts={ntotal}, ep_size={self.ep_size}")
            shard = ntotal // self.ep_size
            self._shard_size = shard
            self._shard_base = self.ep_rank * shard
            self.w13_weights_cpu.append(
                self._allocate_torch_shared_expert_buffer(
                    layer_moe_idx, "w13", w13_shape, params_dtype))
            self.w2_weights_cpu.append(
                self._allocate_torch_shared_expert_buffer(
                    layer_moe_idx, "w2", w2_shape, params_dtype))
        elif self.offload_config.shard_per_rank:
            # shard-per-rank: each rank holds ONLY its EP shard of weight
            # experts (ntotal // ep_size), as per-expert pinned tensors (like
            # the non-shared path, but shard-sized). No mmap, no cross-process
            # sharing, no staging — H2D reads each expert's own pinned storage.
            # Scales/offsets are sharded the same way (see
            # _init_layer_scale_buffers). Placement must
            # be constrained to EP ownership (expert e → rank e // shard) so a
            # rank only loads experts it actually holds.
            shard = ntotal // max(1, self.ep_size)
            self._shard_size = shard
            self._shard_base = self.ep_rank * shard
            w13_list = [
                self._allocate_expert_host_tensor(w13_shape, params_dtype)
                for _ in range(shard)
            ]
            w2_list = [
                self._allocate_expert_host_tensor(w2_shape, params_dtype)
                for _ in range(shard)
            ]
            self.w13_weights_cpu.append(w13_list)
            self.w2_weights_cpu.append(w2_list)
        else:
            w13_list = [
                self._allocate_expert_host_tensor(w13_shape, params_dtype)
                for _ in range(ntotal)
            ]
            w2_list = [
                self._allocate_expert_host_tensor(w2_shape, params_dtype)
                for _ in range(ntotal)
            ]
            self.w13_weights_cpu.append(w13_list)
            self.w2_weights_cpu.append(w2_list)

        # Per-expert storage size (works for both list[0] and big_tensor[0]).
        if self.w13_weights_cpu[-1]:
            first_w13 = self.w13_weights_cpu[-1][0]
            first_w2 = self.w2_weights_cpu[-1][0]
        elif self.exclusive_shared_cpu_enabled:
            first_w13 = self._torch_shared_cpu_buffers[
                (layer_moe_idx, "w13")][0]
            first_w2 = self._torch_shared_cpu_buffers[
                (layer_moe_idx, "w2")][0]
        else:
            raise RuntimeError(
                f"Layer {layer_moe_idx} allocated no CPU expert buffers")
        self.w13_expert_size_bytes = first_w13.nelement() * first_w13.element_size()
        self.w2_expert_size_bytes = first_w2.nelement() * first_w2.element_size()

        # Scale / offset CPU buffers (W8A8)
        self._init_layer_scale_buffers(layer, layer_moe_idx, ntotal)

        self.moe_layers.append(layer)
        # If the cache policy was already built (this layer is registered
        # after _finalize_offload, e.g. an MTP draft MoE layer loaded after
        # the target model), extend the policy and per-layer stats so LRC
        # eviction applies uniformly to target and draft layers. Keeps the
        # invariant that every registered MoE layer has a matching cache
        # state and stats slot.
        self._extend_cache_for_layer()
        # Same post-finalize path for prefetch gate weights: register this
        # layer's gate so _gate_weights_npu stays index-aligned with
        # moe_layers. Without it, len(moe_layers) > len(_gate_weights_npu)
        # and predict_next_layer_experts_npu returns None for the boundary
        # layer. Pre-finalize layers are covered in bulk by
        # register_gate_weights(); the cache_policy sentinel skips them.
        self._register_layer_gate(layer)

    def _extend_cache_for_layer(self):
        """Grow cache_policy and stats lists to cover one more MoE layer.

        No-op before _finalize_offload has built the policy (the target
        layers are all covered in one shot there). Afterwards each newly
        registered layer (e.g. the MTP draft layer) gets its own fresh
        LRC state, so draft-layer hotness is tracked independently from
        the target layers.
        """
        if self.cache_policy is None:
            return
        new_idx = self.cache_policy.add_layer()
        self.cache_requests.append(0)
        self.cache_hits.append(0)
        self.cache_misses.append(0)
        self.cache_calls.append(0)
        self.last_hit_experts.append([])
        self.last_miss_experts.append([])
        logger.info(
            "[EXPERT-OFFLOAD-CACHE] extended cache policy to layer=%d "
            "(total_layers=%d)",
            new_idx, len(self.cache_policy.layer_states))

    @staticmethod
    def _cpu_tensor_storage_bytes(tensors) -> int:
        """Count unique CPU tensor storages in a nested list structure."""
        total = 0
        seen: set[tuple[int, int]] = set()

        def visit(value):
            nonlocal total
            if isinstance(value, torch.Tensor):
                storage = value.untyped_storage()
                key = (storage.data_ptr(), storage.nbytes())
                if key not in seen:
                    seen.add(key)
                    total += storage.nbytes()
            elif isinstance(value, dict):
                for child in value.values():
                    visit(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    visit(child)

        visit(tensors)
        return total

    @staticmethod
    def _host_memory_snapshot() -> tuple[int | None, int | None, int | None]:
        """Return Linux process RSS/HWM and system available bytes."""
        rss = hwm = available = None
        try:
            with open("/proc/self/status", encoding="utf-8") as status_file:
                for line in status_file:
                    if line.startswith("VmRSS:"):
                        rss = int(line.split()[1]) * 1024
                    elif line.startswith("VmHWM:"):
                        hwm = int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass
        try:
            with open("/proc/meminfo", encoding="utf-8") as meminfo_file:
                for line in meminfo_file:
                    if line.startswith("MemAvailable:"):
                        available = int(line.split()[1]) * 1024
                        break
        except (OSError, ValueError, IndexError):
            pass
        return rss, hwm, available

    def _log_cpu_expert_memory(self) -> None:
        """Log per-rank expert buffers and host memory for replica auditing."""
        if not self._debug:
            return
        weights_bytes = self._cpu_tensor_storage_bytes(
            (self.w13_weights_cpu, self.w2_weights_cpu))
        quant_bytes = self._cpu_tensor_storage_bytes((
            self.scale_cpu_buffers,
            self.offset_cpu_buffers,
            self.scale_bias_cpu_buffers,
        ))
        expert_bytes = weights_bytes + quant_bytes
        rss, hwm, available = self._host_memory_snapshot()
        cpu_mode = (
            "exclusive_sharded" if self.exclusive_sharded_cpu_enabled else
            "exclusive_shared" if self.exclusive_shared_cpu_enabled else
            "exclusive_dynamic" if self.exclusive_dynamic_enabled else
            "torch_shared" if self.torch_shared_cpu_weights_enabled else
            "sharded" if self.offload_config.shard_per_rank else
            "replicated")
        experts_per_layer = (
            len(self.w13_weights_cpu[0]) if self.w13_weights_cpu else 0)
        replica_factor = (
            1 if (self.exclusive_dynamic_enabled
                  or self.offload_config.shard_per_rank) else self.ep_size)

        def mib(value):
            return None if value is None else round(value / (1024 ** 2), 1)

        logger.info(
            "[CPU_MEM] rank=%s/%s pid=%s cpu_mode=%s "
            "experts_per_layer=%s global_experts=%s layers=%s "
            "expert_buffers_mib=%.1f weights_mib=%.1f quant_mib=%.1f "
            "process_rss_mib=%s process_hwm_mib=%s host_available_mib=%s "
            "expected_full_weight_copies_across_ep=%s",
            self.ep_rank, self.ep_size, os.getpid(), cpu_mode,
            experts_per_layer, self.num_total_experts, len(self.moe_layers),
            mib(expert_bytes), mib(weights_bytes), mib(quant_bytes),
            mib(rss), mib(hwm), mib(available), replica_factor)

    def _log_torch_shared_cpu_pss(self) -> None:
        """Collectively audit physical pages used by shared expert mmaps."""
        if (not getattr(self, "_debug", False)
                or not self.torch_shared_cpu_weights_enabled):
            return
        pool = self._torch_shared_cpu_pool
        if pool is None:
            return

        from torch import distributed as dist
        from vllm.distributed.parallel_state import get_ep_group

        local_snapshot = pool.mapping_memory_snapshot()
        snapshots = [None] * self.ep_size
        dist.all_gather_object(
            snapshots,
            local_snapshot,
            group=get_ep_group().cpu_group,
        )
        if self.ep_rank != 0:
            return
        if any(snapshot is None for snapshot in snapshots):
            logger.info(
                "[TORCH_SHARED_CPU_PSS] ranks=%d result=UNAVAILABLE "
                "reason=proc_self_smaps_unavailable",
                self.ep_size,
            )
            return

        expected_bytes = self._cpu_tensor_storage_bytes(
            getattr(self, "_torch_shared_cpu_buffers", {}))
        mapped_sizes = [snapshot["size_bytes"] for snapshot in snapshots]
        mapping_counts = [snapshot["mapping_count"]
                          for snapshot in snapshots]
        total_rss_bytes = sum(snapshot["rss_bytes"]
                              for snapshot in snapshots)
        total_pss_bytes = sum(snapshot["pss_bytes"]
                              for snapshot in snapshots)
        total_swap_pss_bytes = sum(snapshot["swap_pss_bytes"]
                                   for snapshot in snapshots)
        pss_ratio = (
            total_pss_bytes / expected_bytes if expected_bytes else 0.0)
        mapped_size_ratios = (
            [size / expected_bytes for size in mapped_sizes]
            if expected_bytes else [0.0] * len(mapped_sizes))
        mappings_complete = (
            expected_bytes > 0
            and all(
                abs(ratio - 1.0)
                <= self._SHARED_CPU_MAPPING_SIZE_TOLERANCE
                for ratio in mapped_size_ratios)
        )
        if not mappings_complete:
            result = "FAIL_MAPPING_SIZE_MISMATCH"
        elif total_swap_pss_bytes:
            result = "INCONCLUSIVE_SWAPPED"
        elif abs(pss_ratio - 1.0) <= self._SHARED_CPU_PSS_RATIO_TOLERANCE:
            result = "PASS_ONE_PHYSICAL_COPY"
        elif pss_ratio > 1.0:
            result = "FAIL_EXTRA_PHYSICAL_PAGES"
        else:
            result = "INCONCLUSIVE_NOT_FULLY_RESIDENT"

        mib = 1024 ** 2
        logger.info(
            "[TORCH_SHARED_CPU_PSS] ranks=%d expected_shared_expert_mib=%.1f "
            "mapped_size_per_rank_mib=%.1f..%.1f "
            "mappings_per_rank=%d..%d mapping_total_rss_mib=%.1f "
            "mapping_total_pss_mib=%.1f mapping_total_swap_pss_mib=%.1f "
            "pss_to_expected_ratio=%.3f result=%s",
            self.ep_size,
            expected_bytes / mib,
            min(mapped_sizes) / mib,
            max(mapped_sizes) / mib,
            min(mapping_counts),
            max(mapping_counts),
            total_rss_bytes / mib,
            total_pss_bytes / mib,
            total_swap_pss_bytes / mib,
            pss_ratio,
            result,
        )

    def _init_layer_scale_buffers(self, layer, layer_moe_idx: int,
                                   ntotal: int):
        """Allocate CPU scale/offset buffers for a single MoE layer."""
        # shard-per-rank: like the weight buffers, each rank holds only its EP
        # shard (ntotal // ep_size) of scale/offset/scale_bias, indexed LOCALLY.
        # global<->local via _shard_local (mirrors the weight path); non-shard
        # (single-card) keeps the full ntotal, global==local.
        use_shard = self.offload_config.shard_per_rank
        if self.exclusive_dynamic_enabled:
            nalloc = len(self._cpu_slot_to_eid[layer_moe_idx])
            if (self.exclusive_sharded_cpu_enabled
                    and layer_moe_idx in getattr(
                        self, "_exclusive_sharded_bootstrap_layers", set())):
                nalloc += len(self._npu_slot_to_eid[layer_moe_idx])
        else:
            nalloc = ((ntotal // max(1, self.ep_size))
                      if use_shard else ntotal)
        attr_specs = [
            ("scale_cpu_buffers", "w13_weight_scale"),
            ("scale_cpu_buffers", "w2_weight_scale"),
            ("offset_cpu_buffers", "w13_weight_offset"),
            ("offset_cpu_buffers", "w2_weight_offset"),
            ("scale_bias_cpu_buffers", "w13_scale_bias"),
            ("scale_bias_cpu_buffers", "w2_scale_bias"),
        ]
        for buffer_dict_name, attr_name in attr_specs:
            if not hasattr(layer, attr_name):
                continue
            dev_tensor = getattr(layer, attr_name)
            dtype = dev_tensor.dtype
            if "scale_bias" in attr_name:
                per_expert_shape = tuple(dev_tensor.shape[1:])
            elif dtype.itemsize == 1:
                from vllm_ascend.quantization.methods.w4a8_mxfp4 import (
                    apply_mxfp4_weight_scale_layout)
                dtype = torch.uint8
                per_expert_shape = tuple(
                    apply_mxfp4_weight_scale_layout(dev_tensor[0].view(torch.uint8)).shape)
            else:
                per_expert_shape = (dev_tensor[0].numel(),)
            buffer_dict: dict = getattr(self, buffer_dict_name)
            if attr_name not in buffer_dict:
                buffer_dict[attr_name] = []
            buffers = buffer_dict[attr_name]
            while len(buffers) <= layer_moe_idx:
                buffers.append([])
            if self.torch_shared_cpu_weights_enabled:
                buffers[layer_moe_idx] = (
                    self._allocate_torch_shared_expert_buffer(
                        layer_moe_idx, attr_name, per_expert_shape, dtype))
                if (self.exclusive_shared_cpu_enabled
                        and layer_moe_idx in getattr(
                            self, "_exclusive_shared_bootstrap_layers", set())):
                    device_slots = (
                        self.offload_config.num_device_experts_for_rank(
                            layer_moe_idx, self.ep_size))
                    buffers[layer_moe_idx].extend(
                        self._allocate_expert_host_tensor(
                            per_expert_shape, dtype)
                        for _ in range(device_slots))
            else:
                for _ in range(nalloc):
                    buffers[layer_moe_idx].append(
                        self._allocate_expert_host_tensor(
                            per_expert_shape, dtype))

    def _finalize_offload(self, model):
        """Post-weight-loading finalization.

        Must be called AFTER get_model() has finished loading all weights.
        Performs NZ format conversion, cache policy init, forward buffer
        init, fp32 scale refresh, prefill pool creation, and gate weight
        registration.
        """
        if not self.moe_layers:
            return
        # Barrier: ensure all deferred load_w13/load_w2/_load_scale_shard
        # copies have landed before process_weights_after_loading reads them.
        self.drain_load_pool()
        t0 = time.perf_counter()
        logger.info(
            "[OFFLOAD] weight load (safetensors→CPU buffer): %.1fs over %d calls",
            self._weight_load_secs, self._weight_load_calls)
        t1 = time.perf_counter()
        self.process_weights_after_loading()
        self._publish_shared_h2d_sources()
        t2 = time.perf_counter()

        num_moe_layers = len(self.moe_layers)
        # Validate a per-layer num_device_experts list covers every MoE layer.
        # Scalars and single-element lists broadcast; a multi-element list is
        # indexed by MoE-layer registration order and must cover at least the
        # layers registered so far. It may be longer to also cover MoE layers
        # that register AFTER _finalize_offload — notably the MTP draft MoE,
        # which loads with the drafter after the target model is finalized, so
        # the count seen here (target layers only) is smaller than the final
        # total. Requiring equality would reject the extra draft-layer entry.
        nde_list = self.offload_config.num_device_experts_list
        self.offload_config.validate_num_moe_layers(num_moe_layers)
        if self._debug:
            logger.info(
                "[OFFLOAD] num_device_experts per layer (n_layers=%d): %s",
                num_moe_layers, nde_list if len(nde_list) > 1 else nde_list[0])
        if self.offload_config.cache_policy_enabled:
            self.cache_requests = [0 for _ in range(num_moe_layers)]
            self.cache_hits = [0 for _ in range(num_moe_layers)]
            self.cache_misses = [0 for _ in range(num_moe_layers)]
            self.cache_calls = [0 for _ in range(num_moe_layers)]
            self.last_hit_experts = [[] for _ in range(num_moe_layers)]
            self.last_miss_experts = [[] for _ in range(num_moe_layers)]
            self.cache_policy = LRCExpertCachePolicy(
                num_layers=num_moe_layers,
                num_experts=self.num_total_experts,
                # cache_size is informational only (LRC eviction keys on
                # hotness, not slot count). Use the representative min; the
                # real per-layer slot count is each layer's device weight size,
                # set per layer via the expert_map_offload.
                cache_size=self.num_device_experts,
                topk=self.topk,
                recent_window=self.offload_config.cache_recent_window,
                ema_beta=self.offload_config.cache_ema_beta,
                recent_weight=self.offload_config.cache_recent_weight,
                ema_weight=self.offload_config.cache_ema_weight,
                router_weight=self.offload_config.cache_router_weight,
                age_weight=self.offload_config.cache_age_weight,
            )
        t3 = time.perf_counter()

        ntotal = self.num_total_experts
        self.topk_ids_h = torch.zeros(
            [self.offload_threshold, self.topk],
            dtype=torch.int32, device="cpu", pin_memory=True)
        self.topk_weights_h = torch.zeros(
            [self.offload_threshold, self.topk],
            dtype=torch.float32, device="cpu", pin_memory=True)
        self.router_logits_h = None
        self.e_score_correction_bias_h = None
        if self.offload_config.expert_substitution_enabled:
            self.router_logits_h = torch.zeros(
                [self.offload_threshold, ntotal],
                dtype=torch.float32, device="cpu", pin_memory=True)
            self.e_score_correction_bias_h = torch.zeros(
                ntotal, dtype=torch.float32, device="cpu", pin_memory=True)
        # Per-rank active-token mask (1=real, 0=pad), mirrored to pinned CPU so
        # the multi-card host callback can drop pad rows before counting. Under
        # single-batch TP, ranks past the real-token count route zero-hidden
        # PAD tokens whose topk is garbage; without this filter they pollute
        # global_counts (placement), the LRU freq, and hit/miss stats.
        self.mc2_mask_h = torch.zeros(
            self.offload_threshold, dtype=torch.int32,
            device="cpu", pin_memory=True)
        self.log2phy_h = torch.zeros(ntotal, dtype=torch.int32,
                                     device='cpu', pin_memory=True)
        self.log2phy_np = self.log2phy_h.numpy()
        if self.exclusive_shared_cpu_enabled:
            self._initialize_exclusive_shared_runtime_state()
            self._log_exclusive_shared_canonical_memory()
        elif self.exclusive_sharded_cpu_enabled:
            self._initialize_exclusive_sharded_runtime_state()
        t4 = time.perf_counter()

        self.refresh_fp32_scales()
        t5 = time.perf_counter()
        self._preload_hot_experts()
        t_hot = time.perf_counter()
        self.create_prefill_pool()
        t6 = time.perf_counter()
        if self.offload_config.prefill_module_prefetch:
            # [dual-offload test] Fill the PrefetchOffloader-owned full-size
            # params from the finalized CPU buffers. Must run BEFORE
            # offloader.post_init() (load_model ordering guarantees this) so
            # param.data is still the prefetch pinned CPU storage.
            self._fill_prefetch_full_params()
        if self.offload_config.expert_prefetch_enabled:
            self.register_gate_weights(model)
            # Pinned staging buffer for graph-mode prefetch: trigger_next_
            # layer_prefetch stages the next layer's log2phy here with
            # non_blocking D2H before launching the host callback, mirroring
            # update_weights (blocking .cpu() on a live graph tensor would
            # deadlock during replay).
            self._prefetch_log2phy_h = torch.zeros(
                self.num_total_experts, dtype=torch.int32,
                device='cpu', pin_memory=True)
            self._prefetch_log2phy_np = self._prefetch_log2phy_h.numpy()
        t7 = time.perf_counter()
        self._log_cpu_expert_memory()
        logger.info(
            "[OFFLOAD] finalize breakdown: process_weights=%.1fs "
            "cache_policy=%.1fs buffers=%.1fs init_device=%.1fs "
            "hot_preload=%.1fs prefill_pool=%.1fs gate=%.1fs | total=%.1fs",
            t2 - t1, t3 - t2, t4 - t3, t5 - t4, t_hot - t5, t6 - t_hot,
            t7 - t6, t7 - t0)

    def process_weights_after_loading(self):
        """Convert resident CPU expert buffers to the on-device weight format.

        W8A8 (int8): fractal NZ cast on the transpose-after buffer layout
        (the device path transposes first, then casts NZ).
        W4A8_DYNAMIC (int8 cpu, int32 device): mirror the device path which
        transposes, casts NZ, then packs 4 int8 → 1 int32.
        W4A8_MXFP (uint8): mirror the device process_weights_after_loading,
        which casts29 (mxfp4) on the *pre-transpose* shape and then
        transposes — so we restore the pre-transpose shape first, cast, and
        transpose back to match the device slot layout byte-for-byte.

        After this runs each w13/w2 CPU tensor still reports its original
        shape but its storage holds on-device-format bytes — a "liar tensor".
        Touch it only via untyped_storage() slicing, never through the tensor
        view. No-op for non-quantized (other dtype) models.
        """
        first_w13 = self.w13_weights_cpu[0][0]
        first_layer = self.moe_layers[0]
        # Detect W4A8_DYNAMIC by scale dtype: process_scale converts float32
        # to int64 on the device for both modelslim and compressed_tensors
        # paths. The weight dtype is NOT a reliable signal — modelslim leaves
        # it as int8 (no pack_to_int32) while compressed_tensors packs to
        # int32. Without this check, modelslim W4A8 would be misrouted to the
        # W8A8 branch, skipping scale encoding and producing garbled output.
        is_w4a8 = (hasattr(first_layer, 'w13_weight_scale') and
                   first_layer.w13_weight_scale.dtype == torch.int64)
        if first_w13.dtype == torch.int8:
            first_dev = _expert_weight(first_layer, "w13_weight")
            if first_dev.dtype == torch.int32:
                # compressed_tensors W4A8: weight packed to int32
                self._cast_cpu_weights_to_device_format(w4a8_dynamic=True)
            else:
                # W8A8 or modelslim W4A8: weight is int8, just NZ cast
                self._cast_cpu_weights_to_device_format(mxfp4=False)
            # Scale encoding for W4A8 (both modelslim and compressed_tensors).
            # Must run AFTER _cast_cpu_weights_to_device_format so the CPU
            # buffers are already in device byte layout.
            if is_w4a8:
                self._process_scale_bias_cpu_buffers()
                self._encode_w4a8_dynamic_weight_scales()
        elif first_w13.dtype == torch.uint8:
            self._cast_cpu_weights_to_device_format(mxfp4=True)
            # W4A8_MXFP: also stamp the on-device expert weight slots with
            # format-29 (NZ) so the format METADATA matches the NZ bytes that
            # _cast_cpu_weights_to_device_format produced in the CPU buffer and
            # that decode/prefill H2D writes into the slots at runtime. Without
            # this the slot stays base-format; the fused-swiglu GMM reads it via
            # raw storage (V4 works), but the SiTU raw npu_grouped_matmul path
            # (Kimi-K3) checks the format and rejects fp4_e2m1 on base format
            # (AclNN EZ1001 / error 161002). Bytes set here are overwritten by
            # H2D; only the format persists.
            self._cast_device_slots_to_mxfp4_nz()
        # else: non-quantized model, no-op
        self._materialize_exclusive_shared_bootstrap()
        self._materialize_exclusive_sharded_bootstrap()

    @staticmethod
    def _tensor_storage_window(
        tensor: torch.Tensor,
        nbytes: int,
    ) -> torch.UntypedStorage:
        """Return this tensor view's byte range within its backing storage.

        ``tensor.untyped_storage()`` returns the complete backing storage and
        does not apply ``tensor.storage_offset()``. That distinction matters
        for one-expert views into the compact Torch shared mmap: copying a
        converted expert into the unsliced storage would target the whole
        shared pool. Private bootstrap tensors have offset zero and use the
        same bounded path.
        """
        storage = tensor.untyped_storage()
        start = tensor.storage_offset() * tensor.element_size()
        end = start + nbytes
        if nbytes <= 0 or start < 0 or end > storage.nbytes():
            raise RuntimeError(
                "Tensor storage window is outside its backing storage: "
                f"offset_bytes={start}, requested_bytes={nbytes}, "
                f"storage_bytes={storage.nbytes()}")
        return storage[start:end]

    def _cast_cpu_weights_to_device_format(self, mxfp4: bool = False,
                                            w4a8_dynamic: bool = False):
        """Relayout resident CPU w13/w2 expert buffers into the device format.

        NZ (W8A8) and format-29 mxfp4 (W4A8_MXFP) are equal-length relayouts,
        so per-expert on-device bytes == nelement * element_size; we still
        recompute from the cast storage to stay correct if that ever changes.

        W4A8_DYNAMIC mirrors the device path: transpose → NZ cast →
        pack 4 int8 into 1 int32 (view as int32).  The storage size is
        preserved (4× fewer elements at 4× element size), so the CPU buffer
        can hold the packed bytes without reallocation.
        """
        num_moe_layers = len(self.w13_weights_cpu)
        use_shard = self.offload_config.shard_per_rank
        for layer_id in range(num_moe_layers):
            num_experts = len(self.w13_weights_cpu[layer_id])
            if num_experts == 0:
                continue
            w13 = torch.stack(self.w13_weights_cpu[layer_id]).to('npu')
            w2 = torch.stack(self.w2_weights_cpu[layer_id]).to('npu')
            if mxfp4:
                w13 = w13.transpose(1, 2).contiguous()
                w2 = w2.transpose(1, 2).contiguous()
                w13 = torch_npu.npu_format_cast(
                    w13.view(torch.uint8), 29,
                    customize_dtype=torch.float8_e4m3fn,
                    input_dtype=torch_npu.float4_e2m1fn_x2,
                )
                w2 = torch_npu.npu_format_cast(
                    w2.view(torch.uint8), 29,
                    customize_dtype=torch.float8_e4m3fn,
                    input_dtype=torch_npu.float4_e2m1fn_x2,
                )
                w13 = w13.transpose(1, 2)
                w2 = w2.transpose(1, 2)
            elif w4a8_dynamic:
                # CPU buffer is already (E, H, dim2) — _copy_w13_shard stored
                # owned.t(), matching the post-transpose device layout. The
                # device path (process_weights_after_loading_modelslim) does
                # transpose(1,2) → NZ cast → pack_to_int32, where transpose
                # converts (E, dim2, H) → (E, H, dim2). Since the CPU buffer
                # is already (E, H, dim2), we NZ-cast directly — NO extra
                # transpose. A double transpose here would apply NZ blocking
                # to (dim2, H) instead of (H, dim2), producing wrong block
                # layout and garbled output.
                w13 = torch_npu.npu_format_cast(w13, ACL_FORMAT_FRACTAL_NZ)
                w2 = torch_npu.npu_format_cast(w2, ACL_FORMAT_FRACTAL_NZ)
                w13 = w13.view(torch.int32)
                w2 = w2.view(torch.int32)
            else:
                w13 = torch_npu.npu_format_cast(w13, ACL_FORMAT_FRACTAL_NZ)
                w2 = torch_npu.npu_format_cast(w2, ACL_FORMAT_FRACTAL_NZ)
            w13_storage = w13.untyped_storage()
            w2_storage = w2.untyped_storage()
            per_w13 = w13_storage.nbytes() // num_experts
            per_w2 = w2_storage.nbytes() // num_experts
            self.w13_expert_size_bytes = per_w13
            self.w2_expert_size_bytes = per_w2
            for local_i in range(num_experts):
                # _expert_dst_storage takes a GLOBAL eid (shard-per-rank
                # remaps it to the local slot); w13/w2_storage are indexed by
                # the local position in the stacked tensor (== global id when
                # not sharded).
                shared_bootstrap = (
                    self.exclusive_shared_cpu_enabled
                    and layer_id in getattr(
                        self, "_exclusive_shared_bootstrap_layers", set()))
                sharded_bootstrap = (
                    self.exclusive_sharded_cpu_enabled
                    and layer_id in getattr(
                        self, "_exclusive_sharded_bootstrap_layers", set()))
                if shared_bootstrap or sharded_bootstrap:
                    # Checkpoint buffers contain compact canonical CPU owners
                    # followed by the initial local NPU owners. Shared-mode
                    # prefix entries are views into the whole mmap, whereas
                    # the suffix and rank-local mode use private tensors. Apply
                    # each tensor view's byte offset before writing either.
                    self._tensor_storage_window(
                        self.w13_weights_cpu[layer_id][local_i],
                        per_w13,
                    ).copy_(
                        w13_storage[local_i * per_w13:
                                    (local_i + 1) * per_w13])
                    self._tensor_storage_window(
                        self.w2_weights_cpu[layer_id][local_i],
                        per_w2,
                    ).copy_(
                        w2_storage[local_i * per_w2:
                                   (local_i + 1) * per_w2])
                    continue
                if self.exclusive_dynamic_enabled:
                    cpu_slot = (
                        self._exclusive_shared_local_cpu_slots[
                            layer_id][local_i]
                        if self.exclusive_shared_cpu_enabled else local_i)
                    geid = self._cpu_slot_to_eid[layer_id][cpu_slot]
                else:
                    geid = ((self._shard_base + local_i)
                            if use_shard else local_i)
                self._expert_dst_storage(layer_id, geid, 'w13').copy_(
                    w13_storage[local_i * per_w13 : (local_i + 1) * per_w13]
                )
                self._expert_dst_storage(layer_id, geid, 'w2').copy_(
                    w2_storage[local_i * per_w2 : (local_i + 1) * per_w2]
                )

    def _materialize_exclusive_shared_bootstrap(self) -> None:
        """Materialize uniformly processed global owners on each local NPU.

        The compact shared mmap contains only CPU-owned experts. During
        checkpoint loading, each rank appends private temporary tensors for the
        global physical slots hosted by that rank. Copy those post-processed
        weights and quant attributes to local NPU slots, synchronize, and drop
        the suffix before publishing shared sources or auditing canonical
        memory. No cross-rank weight transfer is involved.
        """
        bootstrap_layers = getattr(
            self, "_exclusive_shared_bootstrap_layers", set())
        if not self.exclusive_shared_cpu_enabled or not bootstrap_layers:
            return

        for layer_idx in sorted(tuple(bootstrap_layers)):
            layer = self.moe_layers[layer_idx]
            shared_writer_count = len(
                self._exclusive_shared_local_cpu_slots[layer_idx])
            local_npu_count = (
                self.offload_config.num_device_experts_for_rank(
                    layer_idx, self.ep_size))
            expected_count = shared_writer_count + local_npu_count
            if expected_count != self._shard_size:
                raise RuntimeError(
                    "Shared exclusive bootstrap must cover one complete EP "
                    "checkpoint shard: "
                    f"rank={self.ep_rank}, layer={layer_idx}, "
                    f"expected_shard={self._shard_size}, "
                    f"shared_writers={shared_writer_count}, "
                    f"local_npu={local_npu_count}")
            global_slot_base = self.ep_rank * local_npu_count
            local_npu_eids = self._npu_slot_to_eid[layer_idx][
                global_slot_base:global_slot_base + local_npu_count]
            if len(local_npu_eids) != local_npu_count:
                raise RuntimeError(
                    "Shared exclusive bootstrap local NPU owner count mismatch: "
                    f"rank={self.ep_rank}, layer={layer_idx}, "
                    f"expected={local_npu_count}, actual={len(local_npu_eids)}")

            weight_buffers = (
                ("w13", self.w13_weights_cpu[layer_idx],
                 self.w13_expert_size_bytes),
                ("w2", self.w2_weights_cpu[layer_idx],
                 self.w2_expert_size_bytes),
            )
            for name, buffers, _ in weight_buffers:
                if len(buffers) != expected_count:
                    raise RuntimeError(
                        "Shared exclusive bootstrap buffer count mismatch: "
                        f"rank={self.ep_rank}, layer={layer_idx}, name={name}, "
                        f"expected={expected_count}, actual={len(buffers)}")

            tasks = []
            for local_slot, eid in enumerate(local_npu_eids):
                source_index = shared_writer_count + local_slot
                for name, buffers, nbytes in weight_buffers:
                    tasks.append(H2DCopyTask(
                        source=buffers[source_index].untyped_storage(),
                        destination=self._expert_device_storage(
                            layer, local_slot, name),
                        nbytes=nbytes,
                        name=(f"exclusive-shared-bootstrap-{name}"
                              f"[L{layer_idx},E{eid}->S{local_slot}]"),
                    ))

                for buffer_dict in (
                        self.scale_cpu_buffers,
                        self.offset_cpu_buffers,
                        self.scale_bias_cpu_buffers):
                    for attr_name, layer_buffers in buffer_dict.items():
                        if layer_idx >= len(layer_buffers):
                            continue
                        buffers = layer_buffers[layer_idx]
                        dev_tensor = getattr(layer, attr_name, None)
                        if dev_tensor is None:
                            continue
                        if len(buffers) != expected_count:
                            raise RuntimeError(
                                "Shared exclusive bootstrap quant buffer count "
                                "mismatch: "
                                f"rank={self.ep_rank}, layer={layer_idx}, "
                                f"name={attr_name}, expected={expected_count}, "
                                f"actual={len(buffers)}")
                        source = buffers[source_index]
                        destination = dev_tensor.data[local_slot]
                        tasks.append(H2DCopyTask(
                            source=source.reshape(destination.shape),
                            destination=destination,
                            nbytes=(source.numel() * source.element_size()),
                            name=(f"exclusive-shared-bootstrap-{attr_name}"
                                  f"[L{layer_idx},E{eid}->S{local_slot}]"),
                        ))

            with torch_npu.npu.stream(self.load_stream):
                self._get_h2d_transport().copy_batch(tasks)
                for local_slot in range(local_npu_count):
                    self._refresh_expert_fp32_scale(layer, local_slot)
                self._synchronize_h2d()

            # Keep only this rank's writer views into the compact shared mmap.
            # Dropping the private suffix restores the steady-state invariant:
            # compact shared CPU + all local NPU slots == one global model.
            self.w13_weights_cpu[layer_idx] = self.w13_weights_cpu[
                layer_idx][:shared_writer_count]
            self.w2_weights_cpu[layer_idx] = self.w2_weights_cpu[
                layer_idx][:shared_writer_count]
            for buffer_dict in (
                    self.scale_cpu_buffers,
                    self.offset_cpu_buffers,
                    self.scale_bias_cpu_buffers):
                for layer_buffers in buffer_dict.values():
                    if layer_idx < len(layer_buffers):
                        layer_buffers[layer_idx] = layer_buffers[
                            layer_idx][:shared_writer_count]

            bootstrap_layers.remove(layer_idx)
            self._validate_exclusive_ownership(layer_idx)
            logger.info(
                "[EXCLUSIVE-SHARED-BOOTSTRAP] rank=%d/%d layer=%d "
                "checkpoint_local_experts=%d materialized_npu_slots=%d "
                "shared_cpu_writer_slots=%d canonical_shared_cpu_slots=%d "
                "copy_tasks=%d source_layout=cpu_postprocess "
                "cross_rank_weight_transfer=False status=ok",
                self.ep_rank, self.ep_size, layer_idx, expected_count,
                local_npu_count, shared_writer_count,
                len(self._cpu_slot_to_eid[layer_idx]), len(tasks))

    def _cast_device_slots_to_mxfp4_nz(self):
        """Stamp on-device W4A8_MXFP expert weight slots with format-29 (NZ).

        The device slot is already transposed by process_weights (offload
        branch). Mirror the non-offload cast-then-transpose — and the CPU
        relayout in _cast_cpu_weights_to_device_format — by transposing back
        to the original shape, casting 29, then transposing forward again, so
        the slot's format-29 layout is transpose(cast(original)). That is the
        form npu_grouped_matmul's NZ kernel accepts for fp4_e2m1: it infers
        transposeWeight from the layout (EZ1001 if base format, EZ0026 if the
        cast landed on the transposed shape). Slot bytes are refreshed by
        decode/prefill H2D at runtime (CPU buffer holds the matching NZ bytes);
        this call fixes the format + layout metadata.

        Fallback: some NPU devices/drivers only allow base format
        (allow_internal_format=False) and reject the format-29 cast at the ACL
        layer (device error 361001). This call only stamps format *metadata* on
        the resident slots — the bytes themselves are refreshed by decode/prefill
        H2D at runtime, and the fused-swiglu GMM path (DeepSeek-V4) reads them
        via raw storage, so skipping it is safe for V4. The SiTU raw
        npu_grouped_matmul path (Kimi-K3) checks the format and would reject
        fp4_e2m1 on base format (EZ1001); such models are unsupported on these
        devices. On the first cast failure we warn once and return, leaving the
        slots in base format. w.data is reassigned only after a successful cast,
        so a raised exception never leaves a slot half-mutated.
        """
        for layer in self.moe_layers:
            for name in ("w13_weight", "w2_weight"):
                w = _expert_weight(layer, name)
                if w is None:
                    continue
                d = w.data.transpose(1, 2).contiguous()
                try:
                    d = torch_npu.npu_format_cast(
                        d.view(torch.uint8), 29,
                        customize_dtype=torch.float8_e4m3fn,
                        input_dtype=torch_npu.float4_e2m1fn_x2,
                    )
                except RuntimeError as e:
                    logger.warning(
                        "[OFFLOAD] npu_format_cast to format-29 (mxfp4 NZ) "
                        "failed on layer slot %r; leaving expert slots in base "
                        "format. Safe for DeepSeek-V4 (fused-swiglu GMM reads "
                        "raw storage); breaks the Kimi-K3 SiTU "
                        "npu_grouped_matmul path. Error: %s", name, e)
                    return
                w.data = d.transpose(1, 2)

    def _materialize_exclusive_sharded_bootstrap(self) -> None:
        """Materialize uniformly processed initial owners on the local NPU.

        The original routed-expert loader writes initial NPU owners directly to
        their device slots, while CPU-owned experts pass through this manager's
        CPU post-processing.  That is unsafe when a device-wide format cast is
        unsupported: an initial expert evicted by the first exclusive swap can
        otherwise enter the canonical CPU pool with a different byte layout
        from experts loaded from the checkpoint CPU path.

        For every bootstrap layer, the temporary suffix in each CPU buffer is
        ordered exactly like ``_npu_slot_to_eid``.  Copy those already-processed
        weight and quant bytes to the matching local device slots, synchronize,
        then remove the suffix.  No cross-rank transfer or persistent CPU
        duplicate is introduced.
        """
        bootstrap_layers = getattr(
            self, "_exclusive_sharded_bootstrap_layers", set())
        if not self.exclusive_sharded_cpu_enabled or not bootstrap_layers:
            return

        for layer_idx in sorted(tuple(bootstrap_layers)):
            layer = self.moe_layers[layer_idx]
            cpu_count = len(self._cpu_slot_to_eid[layer_idx])
            npu_count = len(self._npu_slot_to_eid[layer_idx])
            expected_count = cpu_count + npu_count
            weight_buffers = (
                ("w13", self.w13_weights_cpu[layer_idx],
                 self.w13_expert_size_bytes),
                ("w2", self.w2_weights_cpu[layer_idx],
                 self.w2_expert_size_bytes),
            )
            for name, buffers, _ in weight_buffers:
                if len(buffers) != expected_count:
                    raise RuntimeError(
                        "Rank-local exclusive bootstrap buffer count mismatch: "
                        f"rank={self.ep_rank}, layer={layer_idx}, name={name}, "
                        f"expected={expected_count}, actual={len(buffers)}")

            tasks = []
            for slot, eid in enumerate(self._npu_slot_to_eid[layer_idx]):
                source_index = cpu_count + slot
                for name, buffers, nbytes in weight_buffers:
                    tasks.append(H2DCopyTask(
                        source=buffers[source_index].untyped_storage(),
                        destination=self._expert_device_storage(
                            layer, slot, name),
                        nbytes=nbytes,
                        name=(f"exclusive-bootstrap-{name}[L{layer_idx},"
                              f"E{eid}->S{slot}]"),
                    ))

                for buffer_dict in (
                        self.scale_cpu_buffers,
                        self.offset_cpu_buffers,
                        self.scale_bias_cpu_buffers):
                    for attr_name, layer_buffers in buffer_dict.items():
                        if layer_idx >= len(layer_buffers):
                            continue
                        buffers = layer_buffers[layer_idx]
                        dev_tensor = getattr(layer, attr_name, None)
                        if dev_tensor is None:
                            continue
                        if len(buffers) != expected_count:
                            raise RuntimeError(
                                "Rank-local exclusive bootstrap quant buffer "
                                "count mismatch: "
                                f"rank={self.ep_rank}, layer={layer_idx}, "
                                f"name={attr_name}, expected={expected_count}, "
                                f"actual={len(buffers)}")
                        source = buffers[source_index]
                        destination = dev_tensor.data[slot]
                        tasks.append(H2DCopyTask(
                            source=source.reshape(destination.shape),
                            destination=destination,
                            nbytes=(source.numel() * source.element_size()),
                            name=(f"exclusive-bootstrap-{attr_name}[L{layer_idx},"
                                  f"E{eid}->S{slot}]"),
                        ))

            with torch_npu.npu.stream(self.load_stream):
                self._get_h2d_transport().copy_batch(tasks)
                for slot in range(npu_count):
                    self._refresh_expert_fp32_scale(layer, slot)
                self._synchronize_h2d()

            # Drop only the temporary initial-NPU suffix.  The retained prefix
            # remains indexed by _cpu_slot_to_eid and is the sole canonical CPU
            # owner set used by runtime swaps.
            self.w13_weights_cpu[layer_idx] = self.w13_weights_cpu[
                layer_idx][:cpu_count]
            self.w2_weights_cpu[layer_idx] = self.w2_weights_cpu[
                layer_idx][:cpu_count]
            for buffer_dict in (
                    self.scale_cpu_buffers,
                    self.offset_cpu_buffers,
                    self.scale_bias_cpu_buffers):
                for layer_buffers in buffer_dict.values():
                    if layer_idx < len(layer_buffers):
                        layer_buffers[layer_idx] = layer_buffers[
                            layer_idx][:cpu_count]

            bootstrap_layers.remove(layer_idx)
            self._validate_exclusive_ownership(layer_idx)
            logger.info(
                "[EXCLUSIVE-SHARDED-BOOTSTRAP] rank=%d/%d layer=%d "
                "checkpoint_local_experts=%d materialized_npu_slots=%d "
                "canonical_cpu_slots=%d copy_tasks=%d source_layout=cpu_postprocess "
                "cross_rank_weight_transfer=False status=ok",
                self.ep_rank, self.ep_size, layer_idx, expected_count,
                npu_count, cpu_count, len(tasks))

    def _process_scale_bias_cpu_buffers(self):
        """Apply update_bias transformation to scale_bias CPU buffers.

        Mirrors the device-side update_bias for W4A8_DYNAMIC new_quant_version:
        w13_scale_bias: (D1, 1) -> transpose -> (1, D1) -> sum(axis=0) -> (D1,)
        w2_scale_bias: (D1, D2) -> transpose -> (D2, D1) -> sum(axis=0) -> (D1,)
        """
        for attr_name, layer_buffers in self.scale_bias_cpu_buffers.items():
            for layer_idx, expert_buffers in enumerate(layer_buffers):
                transformed_buffers = [
                    buf.transpose(0, 1).contiguous().sum(dim=0)
                    for buf in expert_buffers
                ]
                if self.torch_shared_cpu_weights_enabled:
                    template = None
                    if not transformed_buffers:
                        source = self._torch_shared_cpu_buffers[
                            (layer_idx, attr_name)][0]
                        template = source.transpose(
                            0, 1).contiguous().sum(dim=0)
                    new_buffers = self._replace_torch_shared_expert_buffer(
                        layer_idx, attr_name, transformed_buffers, template)
                else:
                    new_buffers = []
                    for transformed in transformed_buffers:
                        backend_buffer = self._allocate_expert_host_tensor(
                            transformed.shape, transformed.dtype)
                        backend_buffer.copy_(transformed)
                        new_buffers.append(backend_buffer)
                layer_buffers[layer_idx] = new_buffers

    def _encode_w4a8_dynamic_weight_scales(self):
        """Encode W4A8_DYNAMIC weight_scale CPU buffers to device int64 format.

        The safetensors checkpoint stores ``w13_weight_scale`` /
        ``w2_weight_scale`` as float32 tensors, but the device-side
        ``AscendW4A8DynamicFusedMoEMethod.process_scale`` reinterprets the
        float32 bytes as uint32 and zero-extends to int64 before storing it
        on the NPU. The decode-path H2D ``copy_`` therefore must write
        int64-encoded bytes — copying raw float32 into an int64 device tensor
        would corrupt the kernel's scale decoding.

        This method mirrors the per-channel branch of ``process_scale``
        (the only branch supported by expert offload today): float32 →
        uint32 bit-reinterpret → int64 zero-extension. Each expert buffer is
        encoded independently (per-channel encoding is element-wise), so the
        transformation is applied per-expert without cross-expert ops.

        After this runs the CPU buffer dtype changes from float32 to int64,
        matching ``layer.w13_weight_scale.dtype`` on the NPU.
        """
        import numpy as np
        for attr_name in ("w13_weight_scale", "w2_weight_scale"):
            if attr_name not in self.scale_cpu_buffers:
                continue
            for layer_idx, expert_buffers in enumerate(
                    self.scale_cpu_buffers[attr_name]):
                encoded_values = []
                for buf in expert_buffers:
                    # buf: float32, shape per-expert (e.g. (2*IN,) for w13)
                    scale_np = np.ascontiguousarray(
                        buf.cpu().numpy()).astype(np.float32)
                    # Bit-reinterpret float32 bytes as uint32, then
                    # zero-extend to int64 — identical to device process_scale
                    # per-channel branch.
                    scale_np.dtype = np.uint32
                    encoded = scale_np.astype(np.int64)
                    encoded_tensor = torch.from_numpy(np.ascontiguousarray(
                        encoded.copy()))
                    encoded_values.append(encoded_tensor)
                if self.torch_shared_cpu_weights_enabled:
                    template = None
                    if not encoded_values:
                        source = self._torch_shared_cpu_buffers[
                            (layer_idx, attr_name)][0]
                        source_np = np.ascontiguousarray(
                            source.cpu().numpy()).astype(np.float32)
                        source_np.dtype = np.uint32
                        template = torch.from_numpy(np.ascontiguousarray(
                            source_np.astype(np.int64).copy()))
                    encoded_buffers = self._replace_torch_shared_expert_buffer(
                        layer_idx, attr_name, encoded_values, template)
                else:
                    encoded_buffers = []
                    for encoded_tensor in encoded_values:
                        encoded_buf = self._allocate_expert_host_tensor(
                            encoded_tensor.shape, encoded_tensor.dtype)
                        encoded_buf.copy_(encoded_tensor)
                        encoded_buffers.append(encoded_buf)
                self.scale_cpu_buffers[attr_name][layer_idx] = encoded_buffers

    # ------------------------------------------------------------------ #
    #  Deferred weight-load pool                                          #
    # ------------------------------------------------------------------ #
    #
    # Weight loading is callback-driven: the safetensors loader calls
    # load_w13/load_w2/_load_scale_shard once per shard (~99k calls), serially
    # in the main thread. The per-call strided transpose-copy into pinned
    # memory is ~0.2 GB/s single-threaded, which dominated startup (~9 min).
    #
    # Strategy: each loader callback (a) owns the shard via a synchronous
    # .clone() while the safetensors mmap is still mapped, then (b) submits
    # the strided transpose-copy to a worker pool and returns immediately.
    # The main thread keeps pulling shards while the pool churns through
    # copies concurrently. drain_load_pool() barriers before _finalize_offload
    # reads the buffers. Because the deferred copy reads the owned clone (not
    # the mmap view), it stays correct after the safetensors mmap is unmapped
    # (which happens before _finalize_offload runs).

    def _get_load_pool(self) -> ThreadPoolExecutor:
        if self._load_pool is None:
            # Pin torch intra-op threads to 1: otherwise each copy_ spawns
            # nproc libgomp threads and 32 workers x 640 cores exhausts the
            # thread limit (EAGAIN). Parallelism comes from the pool itself.
            self._saved_num_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            self._load_pool = ThreadPoolExecutor(
                max_workers=self._LOAD_POOL_WORKERS,
                thread_name_prefix="offload-load")
            self._load_phase_start = time.perf_counter()
            logger.info(
                "[OFFLOAD] starting parallel weight load (workers=%d)",
                self._LOAD_POOL_WORKERS)
        return self._load_pool

    def _track_load_future(self, fut) -> None:
        self._load_futures.append(fut)
        if len(self._load_futures) >= self._LOAD_POOL_DRAIN_EVERY:
            self._drain_futures()

    def _drain_futures(self) -> None:
        if not self._load_futures:
            return
        # f.result() re-raises any worker exception (e.g. shape mismatch).
        n = len(self._load_futures)
        for f in self._load_futures:
            f.result()
        self._load_futures.clear()
        previously_drained = getattr(self, "_drained_shards", 0)
        self._drained_shards = previously_drained + n
        t0 = getattr(self, "_load_phase_start", None)
        crossed_progress_boundary = (
            self._drained_shards // self._LOAD_PROGRESS_LOG_EVERY
            > previously_drained // self._LOAD_PROGRESS_LOG_EVERY
        )
        if t0 and crossed_progress_boundary:
            elapsed = time.perf_counter() - t0
            logger.info(
                "[OFFLOAD] weight load progress: %d shards copied "
                "(%.0f shards/s, %.0fs elapsed)",
                self._drained_shards,
                self._drained_shards / max(elapsed, 1e-6),
                elapsed,
            )

    def drain_load_pool(self) -> None:
        """Wait for all deferred weight copies to finish.

        Safe to call after the safetensors mmap is unmapped: deferred copies
        read owned clones, not mmap views.
        """
        self._drain_futures()
        if self._load_pool is not None:
            self._load_pool.shutdown(wait=True)
            self._load_pool = None
            if self._saved_num_threads is not None:
                torch.set_num_threads(self._saved_num_threads)
                self._saved_num_threads = None
            self._weight_load_secs = time.perf_counter() - self._load_phase_start

    # -- int4 packing helper for W4A8_DYNAMIC checkpoints -- #

    @staticmethod
    def _pack_int4_dim0(weight: torch.Tensor) -> torch.Tensor:
        """Pack pairs of int4 values along dim 0.

        W4A8_DYNAMIC (msModelSlim new_quant_version) checkpoint weights store
        one int4 value per int8 element along the output dimension.  The
        device tensor expects two int4 values packed into one int8 byte,
        halving dim 0.  This helper performs that packing.

        For w1/w3 shard ``(IN, H)`` -> ``(IN // 2, H)``.
        For w2 ``(H, IN)`` -> ``(H // 2, IN)``.
        """
        if weight.dtype != torch.int8:
            weight = weight.to(torch.int8)
        assert weight.shape[0] % 2 == 0, (
            f"dim 0 must be even for int4 packing, got {weight.shape[0]}")
        pairs = weight.reshape(weight.shape[0] // 2, 2, *weight.shape[1:])
        lo = pairs[:, 0] & 0x0F
        hi = pairs[:, 1] & 0x0F
        return ((hi << 4) | lo).contiguous()

    # -- worker copy kernels (static: no self, no shared mutable state) -- #

    @staticmethod
    def _copy_w13_shard(cpu: torch.Tensor, owned: torch.Tensor,
                        shard_id: str, intermed: int) -> None:
        if shard_id == "w1":
            cpu[:, :intermed].copy_(owned.t())
        elif shard_id == "w3":
            cpu[:, intermed: intermed + owned.shape[0]].copy_(owned.t())

    @staticmethod
    def _copy_w2(dst: torch.Tensor, owned: torch.Tensor) -> None:
        dst.copy_(owned.t())

    @staticmethod
    def _copy_scale_assembled(target: torch.Tensor,
                              w1: torch.Tensor, w3: torch.Tensor) -> None:
        assembled = torch.cat([w1, w3], dim=0)
        if target.dtype == torch.uint8:
            # W4A8_MXFP: store the post-layout bytes so the 1D buffer matches
            # the post-process device slot element order (the device path
            # applies reshape(...,k//2,2).transpose to the e8m0 scale).
            from vllm_ascend.quantization.methods.w4a8_mxfp4 import (
                apply_mxfp4_weight_scale_layout)
            assembled = apply_mxfp4_weight_scale_layout(assembled.view(torch.uint8))
        target.copy_(assembled.reshape(target.shape))

    @staticmethod
    def _copy_scale_direct(target: torch.Tensor, owned: torch.Tensor) -> None:
        if target.dtype == torch.uint8:
            from vllm_ascend.quantization.methods.w4a8_mxfp4 import (
                apply_mxfp4_weight_scale_layout)
            owned = apply_mxfp4_weight_scale_layout(owned.view(torch.uint8))
        target.copy_(owned.reshape(target.shape))

    # ------------------------------------------------------------------ #
    #  Weight-load entry points (called by the safetensors loader)        #
    # ------------------------------------------------------------------ #

    def register_gate_weights(self, _model):
        """Store an fp32 NPU copy of gate.weight for each MoE layer.

        Called from _finalize_offload() after all MoE layers are registered.
        Used by predict_next_layer_experts_npu() so prediction runs on-device
        and can be captured in a CUDA/NPU graph.
        """
        # moe_layers is the authoritative registration order used by every
        # per-layer offload array. Its entries are RoutedExperts objects, so
        # the runner propagates the owning model gate onto each entry. This is
        # model-agnostic (DeepSeek and Kimi K3 use different wrapper classes)
        # and keeps missing gates represented by None rather than shifting all
        # later layer indices.
        self._gate_weights_npu = []
        for layer in self.moe_layers:
            gate = getattr(layer, "gate", None)
            gate_param = getattr(gate, "weight", None)
            self._gate_weights_npu.append(
                None if gate_param is None else gate_param.data.float().clone())
        logger.info("[PREFETCH] registered gate weights for %d MoE layers",
                    len(self._gate_weights_npu))

    def _register_layer_gate(self, layer):
        """Stage one MoE layer's gate.weight for prefetch prediction.

        Single-layer counterpart to register_gate_weights(), for layers
        registered after _finalize_offload (e.g. the MTP draft MoE). Keeps
        _gate_weights_npu index-aligned with moe_layers so
        predict_next_layer_experts_npu can look up
        _gate_weights_npu[next_idx] for every registered layer.

        No-op before _finalize_offload has built the runtime buffers (the
        target layers are covered in bulk there). A missing gate is appended
        as None to preserve index alignment.
        """
        if not self.offload_config.expert_prefetch_enabled:
            return
        if not hasattr(self, "topk_ids_h"):
            return
        gate = getattr(layer, 'gate', None)
        gate_param = getattr(gate, 'weight', None)
        self._gate_weights_npu.append(
            None if gate_param is None else gate_param.data.float().clone())
        logger.info(
            "[PREFETCH] registered gate weight for post-finalize layer "
            "(total gates=%d, moe_layers=%d)",
            len(self._gate_weights_npu), len(self.moe_layers))

    def _shard_local(self, global_eid: int) -> int | None:
        """Map a GLOBAL expert id to the CPU weight buffer's local index.

        shard-per-rank: this rank owns shard [base, base+shard); return the
        local index, or None if the expert isn't owned here (caller skips the
        load). Other modes: identity — the buffer is full (global-indexed) or
        shared (global-indexed mmap slice).
        """
        if not self.offload_config.shard_per_rank:
            return global_eid
        local = global_eid - self._shard_base
        return local if 0 <= local < self._shard_size else None

    def _cpu_local(self, layer_idx: int, global_eid: int) -> int | None:
        """Resolve an expert's current compact CPU slot, if CPU-owned."""
        if self.exclusive_dynamic_enabled:
            slot = self._eid_to_cpu_slot[layer_idx][global_eid]
            return None if slot < 0 else slot
        return self._shard_local(global_eid)

    def _checkpoint_cpu_local(
        self, layer_idx: int, global_eid: int
    ) -> int | None:
        """Resolve this rank's writable checkpoint slot for one CPU expert."""
        cpu_slot = self._cpu_local(layer_idx, global_eid)
        if (cpu_slot is None and self.exclusive_shared_cpu_enabled
                and layer_idx in getattr(
                    self, "_exclusive_shared_bootstrap_layers", set())):
            physical_slot = self._eid_to_npu_slot[layer_idx][global_eid]
            local_npu_count = (
                self.offload_config.num_device_experts_for_rank(
                    layer_idx, self.ep_size))
            global_slot_base = self.ep_rank * local_npu_count
            if (global_slot_base <= physical_slot
                    < global_slot_base + local_npu_count):
                return (
                    len(self._exclusive_shared_local_cpu_slots[layer_idx])
                    + physical_slot - global_slot_base)
        if (cpu_slot is None and self.exclusive_sharded_cpu_enabled
                and layer_idx in getattr(
                    self, "_exclusive_sharded_bootstrap_layers", set())):
            npu_slot = self._eid_to_npu_slot[layer_idx][global_eid]
            if npu_slot >= 0:
                cpu_slot = len(self._cpu_slot_to_eid[layer_idx]) + npu_slot
        if cpu_slot is None or not self.exclusive_shared_cpu_enabled:
            return cpu_slot
        return self._exclusive_shared_cpu_slot_to_local[layer_idx].get(
            cpu_slot)

    def load_w13(self, layer_moe_idx: int, expert_id: int,
                 loaded_weight: torch.Tensor, shard_id: str):
        """Store w1/w3 shard to CPU buffer (transposed) via the load pool."""
        self._weight_load_calls += 1
        idx = self._checkpoint_cpu_local(layer_moe_idx, expert_id)
        if idx is None:
            return  # sharded remote expert or exclusive NPU-owned expert
        cpu = self.w13_weights_cpu[layer_moe_idx][idx]
        intermed = cpu.shape[1] // 2
        if loaded_weight.ndim > 0 and loaded_weight.shape[0] > intermed:
            if loaded_weight.shape[0] == 2 * intermed:
                loaded_weight = self._pack_int4_dim0(loaded_weight)
            else:
                loaded_weight = loaded_weight.narrow(0, 0, intermed)
        owned = loaded_weight.cpu().clone()
        fut = self._get_load_pool().submit(
            self._copy_w13_shard, cpu, owned, shard_id, intermed)
        self._track_load_future(fut)

    def load_w2(self, layer_moe_idx: int, expert_id: int,
                loaded_weight: torch.Tensor):
        """Store w2 weight to CPU buffer (transposed) via the load pool."""
        self._weight_load_calls += 1
        idx = self._checkpoint_cpu_local(layer_moe_idx, expert_id)
        if idx is None:
            return  # sharded remote expert or exclusive NPU-owned expert
        dst = self.w2_weights_cpu[layer_moe_idx][idx]
        owned = loaded_weight.cpu().clone()
        fut = self._get_load_pool().submit(self._copy_w2, dst, owned)
        self._track_load_future(fut)

    # ------------------------------------------------------------------ #
    #  Scale / offset helpers (quantized models only)                     #
    # ------------------------------------------------------------------ #

    def _load_scale_shard(self, layer_moe_idx: int, expert_id: int,
                          attr_name: str, shard_id: str,
                          loaded_weight: torch.Tensor):
        """Load a scale/offset shard into its CPU buffer via the load pool.

        w13 scale/offset arrives as two shards (w1, w3) that must be
        concatenated along dim 0. We stash the first-arriving owned clone in
        _scale_shard_temp and assemble when the second shard arrives.
        w2 scale/offset is a single shard — clone and defer directly.
        """
        self._weight_load_calls += 1
        assert shard_id in ("w1", "w2", "w3"), f"unexpected shard_id: {shard_id}"
        if "scale_bias" in attr_name:
            target_dict = self.scale_bias_cpu_buffers
        elif "scale" in attr_name:
            target_dict = self.scale_cpu_buffers
        else:
            target_dict = self.offset_cpu_buffers
        # shard-per-rank: skip scales for non-owned experts and index the
        # shard-sized buffer locally (mirrors load_w13/load_w2).
        local_eid = self._checkpoint_cpu_local(layer_moe_idx, expert_id)
        if local_eid is None:
            return  # sharded remote expert or exclusive NPU-owned expert
        target = target_dict[attr_name][layer_moe_idx][local_eid]
        if attr_name.startswith("w13_"):
            key = (layer_moe_idx, expert_id, attr_name)
            pending_shard = self._scale_shard_temp.pop(key, None)
            if pending_shard is not None:
                # Second shard — own it, then defer cat + copy.
                cur_shard = loaded_weight.cpu().clone()
                if shard_id == "w1":
                    w1, w3 = cur_shard, pending_shard
                else:
                    w1, w3 = pending_shard, cur_shard
                fut = self._get_load_pool().submit(
                    self._copy_scale_assembled, target, w1, w3)
                self._track_load_future(fut)
            else:
                # First shard — stash an owned clone.
                self._scale_shard_temp[key] = loaded_weight.cpu().clone()
        else:
            # w2 scale/offset — single shard.
            owned = loaded_weight.cpu().clone()
            fut = self._get_load_pool().submit(
                self._copy_scale_direct, target, owned)
            self._track_load_future(fut)

    def refresh_fp32_scales(self):
        """Recompute the derived fp32 per-expert scale after weight loading.

        Device experts are already in place (loaded by the weight loader and
        process_weights_after_loading); this only refreshes
        w13_weight_scale_fp32 from the freshly-loaded w13_weight_scale.
        """
        for i, layer in enumerate(self.moe_layers):
            ndev = min(self.num_device_experts_for_layer(i),
                       _expert_weight(layer, "w13_weight").shape[0])
            if hasattr(layer, 'w13_weight_scale_fp32'):
                for j in range(ndev):
                    layer.w13_weight_scale_fp32[j].copy_(
                        layer.w13_weight_scale.data[j].to(torch.float32))

    def create_prefill_pool(self):
        """Allocate prefill pool tensors on NPU with full expert count.

        Called from _finalize_offload() after decode buffers are set up.
        Creates ndl device tensors each holding all experts (e.g. 128).
        These are used when num_tokens > offload_threshold (large-batch
        prefill), loaded via full-overwrite in _prefill_load_layer.
        """
        if self._prefill_initialized:
            return
        if not self.moe_layers:
            return
        ndl = self.num_device_layers
        pool_layer = self.moe_layers[0]
        _pool_w13 = _expert_weight(pool_layer, "w13_weight")
        dev = _pool_w13.device
        dt = _pool_w13.dtype
        # Size the pool to the per-rank EP shard, NOT the global expert count.
        # The All2All prefill GMM (aclnnGroupedMatmulWeightNz) requires
        # groupList == weight.dim0; groupList = shard, so the pool must hold
        # exactly `shard` experts per rank. mc_shard_size == num_total//ep_size,
        # which is ntotal for single-card (ep_size=1) — so single-card is
        # unchanged (pool holds all experts), multi-card holds the rank's shard.
        ntotal = self.mc_shard_size

        for _ in range(ndl):
            self._alloc_prefill_pool_slot(pool_layer, dev, dt, ntotal)

        # Cast prefill pool weight tensors to the on-device format (kernel
        # requires it). Must happen BEFORE loading data — same order as decode
        # path: create → format-cast → copy_(cpu → npu).
        self._cast_prefill_pool_format(dev, dt)

        # Prefill log2phy: identity — all experts mapped to their slots
        self._prefill_log2phy = torch.arange(ntotal, dtype=torch.int32, device=dev)

        if self.enable_multi_card:
            from vllm_ascend.expert_offload.multi_card_planner import (
                all2all_expert_ids_per_ep_rank,
                all2all_local_expert_indices,
            )

            self._prefill_local_expert_indices = all2all_local_expert_indices(
                self.ep_rank, ntotal)
            self._prefill_expert_ids_per_ep_rank = torch.tensor(
                all2all_expert_ids_per_ep_rank(
                    ntotal, self.num_total_experts),
                dtype=torch.int32,
                device=dev,
            )

        # Pre-initialize all pool slots with layer 0 weights so that
        # profile_run / _dummy_run (which may use prefill path) has
        # valid data.  Subsequent _prefill_load_layer calls will
        # overwrite with the correct per-layer weights.
        self._init_prefill_pool_data(dev, ntotal, ndl)
        self._prefill_initialized = True
        logger.info("[PREFILL_POOL] allocated %d layers × %d experts, "
                    "w13[0].shape=%s w2[0].shape=%s",
                    ndl, ntotal,
                    tuple(self._prefill_w13[0].shape),
                    tuple(self._prefill_w2[0].shape))

    def _alloc_prefill_pool_slot(self, pool_layer, dev, dt, ntotal: int):
        """Append one prefill-pool slot (weights always; scales/offsets/scale_bias
        only if the layer carries them). Weights use the layer dtype `dt`;
        per-channel fp32 scales use float32; the rest use their source dtype."""
        # (target_attr, source_attr, dtype_override)
        quant_specs = [
            ("_prefill_w13_scale", "w13_weight_scale", None),
            ("_prefill_w13_scale_fp32", "w13_weight_scale_fp32", torch.float32),
            ("_prefill_w13_offset", "w13_weight_offset", None),
            ("_prefill_w2_scale", "w2_weight_scale", None),
            ("_prefill_w2_offset", "w2_weight_offset", None),
            ("_prefill_w13_scale_bias", "w13_scale_bias", None),
            ("_prefill_w2_scale_bias", "w2_scale_bias", None),
        ]
        self._prefill_w13.append(torch.empty(
            (ntotal,) + tuple(_expert_weight(pool_layer, "w13_weight").shape[1:]), dtype=dt, device=dev))
        self._prefill_w2.append(torch.empty(
            (ntotal,) + tuple(_expert_weight(pool_layer, "w2_weight").shape[1:]), dtype=dt, device=dev))
        for tgt, src, dtype_override in quant_specs:
            if not hasattr(pool_layer, src):
                continue
            src_t = getattr(pool_layer, src)
            dtype = dtype_override if dtype_override is not None else src_t.dtype
            getattr(self, tgt).append(torch.empty(
                (ntotal,) + tuple(src_t.shape[1:]), dtype=dtype, device=dev))

    def _cast_prefill_pool_format(self, dev, dt):
        """Cast prefill-pool weight tensors to the on-device (kernel) format.

        Must run BEFORE data is loaded (same create → format-cast ordering as
        the decode path). dtype-dispatched:
          - int8 (W8A8): straight FRACTAL_NZ cast.
          - int32 (W4A8_DYNAMIC): rebuild via int8 backing → NZ → view int32
            (an empty int32 tensor can't be NZ-cast directly).
          - uint8 (W4A8_MXFP): cast29 on the pre-transpose shape, then transpose.
        """
        n = len(self._prefill_w13)
        if dt == torch.int8:
            from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ
            for i in range(n):
                self._prefill_w13[i] = torch_npu.npu_format_cast(
                    self._prefill_w13[i], ACL_FORMAT_FRACTAL_NZ)
                self._prefill_w2[i] = torch_npu.npu_format_cast(
                    self._prefill_w2[i], ACL_FORMAT_FRACTAL_NZ)
        elif dt == torch.int32:
            from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ
            for i in range(n):
                t13 = self._prefill_w13[i]
                t2 = self._prefill_w2[i]
                t13_nz = torch_npu.npu_format_cast(
                    torch.empty(t13.shape[:-1] + (t13.shape[-1] * 4,),
                                dtype=torch.int8, device=dev),
                    ACL_FORMAT_FRACTAL_NZ)
                t2_nz = torch_npu.npu_format_cast(
                    torch.empty(t2.shape[:-1] + (t2.shape[-1] * 4,),
                                dtype=torch.int8, device=dev),
                    ACL_FORMAT_FRACTAL_NZ)
                self._prefill_w13[i] = t13_nz.view(torch.int32)
                self._prefill_w2[i] = t2_nz.view(torch.int32)
        elif dt == torch.uint8:
            for i in range(n):
                for attr in ("_prefill_w13", "_prefill_w2"):
                    t = getattr(self, attr)[i]
                    t = torch_npu.npu_format_cast(
                        t.transpose(1, 2).contiguous().view(torch.uint8), 29,
                        customize_dtype=torch.float8_e4m3fn,
                        input_dtype=torch_npu.float4_e2m1fn_x2,
                    )
                    getattr(self, attr)[i] = t.transpose(1, 2)

    def _build_prefill_h2d_tasks(
        self,
        layer_idx: int,
        pool_slot: int,
        source_eid: int,
        destination_eid: int,
    ) -> list[H2DCopyTask]:
        """Build mixed-source copies for one prefill-pool expert.

        Legacy ownership always uses CPU H2D.  Dynamic exclusive ownership
        resolves the current canonical source per expert: CPU residents use
        H2D and NPU residents use D2D from their decode slot.  Prefill does not
        mutate ownership; it remains a full overwrite of the prefill pool.
        """
        w13_start = destination_eid * self.w13_expert_size_bytes
        w2_start = destination_eid * self.w2_expert_size_bytes
        w13_dst = self._prefill_w13[pool_slot].untyped_storage()[
            w13_start:w13_start + self.w13_expert_size_bytes]
        w2_dst = self._prefill_w2[pool_slot].untyped_storage()[
            w2_start:w2_start + self.w2_expert_size_bytes]
        npu_slot = -1
        direction = CopyDirection.H2D
        if self.exclusive_dynamic_enabled:
            npu_slot = self._eid_to_npu_slot[layer_idx][source_eid]
        if npu_slot >= 0:
            source_layer = self.moe_layers[layer_idx]
            w13_source = self._expert_device_storage(
                source_layer, npu_slot, "w13")
            w2_source = self._expert_device_storage(
                source_layer, npu_slot, "w2")
            direction = CopyDirection.D2D
        else:
            w13_source = self._expert_src_storage(
                layer_idx, source_eid, 'w13')
            w2_source = self._expert_src_storage(
                layer_idx, source_eid, 'w2')

        tasks = [
            H2DCopyTask(
                source=w13_source,
                destination=w13_dst,
                nbytes=self.w13_expert_size_bytes,
                name=(f"prefill-w13[L{layer_idx},E{source_eid}"
                      f"->P{pool_slot}:E{destination_eid}]"),
                direction=direction,
            ),
            H2DCopyTask(
                source=w2_source,
                destination=w2_dst,
                nbytes=self.w2_expert_size_bytes,
                name=(f"prefill-w2[L{layer_idx},E{source_eid}"
                      f"->P{pool_slot}:E{destination_eid}]"),
                direction=direction,
            ),
        ]

        local_eid = self._cpu_local(layer_idx, source_eid)
        quant_specs = (
            (self.scale_cpu_buffers, "w13_weight_scale",
             self._prefill_w13_scale),
            (self.scale_cpu_buffers, "w2_weight_scale",
             self._prefill_w2_scale),
            (self.offset_cpu_buffers, "w13_weight_offset",
             self._prefill_w13_offset),
            (self.offset_cpu_buffers, "w2_weight_offset",
             self._prefill_w2_offset),
            (self.scale_bias_cpu_buffers, "w13_scale_bias",
             self._prefill_w13_scale_bias),
            (self.scale_bias_cpu_buffers, "w2_scale_bias",
             self._prefill_w2_scale_bias),
        )
        for cpu_buffers, attr_name, prefill_buffers in quant_specs:
            if (pool_slot >= len(prefill_buffers)
                    or attr_name not in cpu_buffers
                    or layer_idx >= len(cpu_buffers[attr_name])):
                continue
            dst = prefill_buffers[pool_slot][destination_eid]
            if npu_slot >= 0:
                source_layer = self.moe_layers[layer_idx]
                source = getattr(source_layer, attr_name).data[npu_slot]
                nbytes = source.numel() * source.element_size()
                attr_direction = CopyDirection.D2D
            elif (not self.exclusive_shared_cpu_enabled
                    and local_eid is not None
                    and local_eid < len(cpu_buffers[attr_name][layer_idx])):
                src_tensor = cpu_buffers[attr_name][layer_idx][local_eid]
                source = src_tensor.reshape(dst.shape)
                nbytes = src_tensor.numel() * src_tensor.element_size()
                attr_direction = CopyDirection.H2D
            else:
                source = self._shared_h2d_source(
                    layer_idx, source_eid, attr_name)
                nbytes = dst.numel() * dst.element_size()
                attr_direction = CopyDirection.H2D
            tasks.append(H2DCopyTask(
                source=source,
                destination=dst,
                nbytes=nbytes,
                name=(f"prefill-{attr_name}[L{layer_idx},E{source_eid}"
                      f"->P{pool_slot}:E{destination_eid}]"),
                direction=attr_direction,
            ))
        return tasks

    def _refresh_prefill_fp32_scale(self, pool_slot: int,
                                    num_experts: int) -> None:
        if (pool_slot >= len(self._prefill_w13_scale_fp32)
                or pool_slot >= len(self._prefill_w13_scale)):
            return
        count = min(num_experts,
                    self._prefill_w13_scale[pool_slot].shape[0])
        for eid in range(count):
            self._prefill_w13_scale_fp32[pool_slot][eid].copy_(
                self._prefill_w13_scale[pool_slot][eid].to(torch.float32),
                non_blocking=True)

    def _build_prefill_local_d2d_tasks(
        self,
        layer_idx: int,
        pool_slot: int,
        source_slot: int,
        destination_slot: int,
        expert_id: int,
    ) -> list[H2DCopyTask]:
        """Build same-rank prefill copies without typed weight ``copy_``.

        The 950DT aclnn-only runtime rejects ``Tensor.copy_`` when the weight
        tensor carries an internal format.  The existing single-card path
        avoids that dispatch by copying the raw weight storages.  Keep the
        same representation for same-rank multi-card D2D; quantization
        metadata remains ordinary-format typed tensors.
        """
        layer = self.moe_layers[layer_idx]
        tasks = []
        weight_specs = (
            ("w13", self._prefill_w13, self.w13_expert_size_bytes),
            ("w2", self._prefill_w2, self.w2_expert_size_bytes),
        )
        for name, prefill_buffers, nbytes in weight_specs:
            start = destination_slot * nbytes
            source = self._expert_device_storage(layer, source_slot, name)
            destination = prefill_buffers[pool_slot].untyped_storage()[
                start:start + nbytes]
            tasks.append(H2DCopyTask(
                source=source,
                destination=destination,
                nbytes=nbytes,
                name=(f"prefill-local-d2d-{name}[L{layer_idx},"
                      f"E{expert_id},S{source_slot}->P{pool_slot}:"
                      f"E{destination_slot}]"),
                direction=CopyDirection.D2D,
            ))

        for name, source, destination in self._prefill_device_components(
                layer_idx, pool_slot, source_slot, destination_slot,
                include_weights=False):
            tasks.append(H2DCopyTask(
                source=source,
                destination=destination,
                nbytes=source.numel() * source.element_size(),
                name=(f"prefill-local-d2d-{name}[L{layer_idx},"
                      f"E{expert_id},S{source_slot}->P{pool_slot}:"
                      f"E{destination_slot}]"),
                direction=CopyDirection.D2D,
            ))
        return tasks

    def _prefill_device_components(
        self,
        layer_idx: int,
        pool_slot: int,
        source_slot: int,
        destination_slot: int,
        *,
        include_weights: bool = True,
    ) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
        """Return matching typed views for local or buffered prefill copies."""
        layer = self.moe_layers[layer_idx]
        components = []
        if include_weights:
            components.extend([
                ("w13", _expert_weight(layer, "w13_weight").data[source_slot],
                 self._prefill_w13[pool_slot][destination_slot]),
                ("w2", _expert_weight(layer, "w2_weight").data[source_slot],
                 self._prefill_w2[pool_slot][destination_slot]),
            ])
        quant_specs = (
            ("w13_weight_scale", self._prefill_w13_scale),
            ("w2_weight_scale", self._prefill_w2_scale),
            ("w13_weight_offset", self._prefill_w13_offset),
            ("w2_weight_offset", self._prefill_w2_offset),
            ("w13_scale_bias", self._prefill_w13_scale_bias),
            ("w2_scale_bias", self._prefill_w2_scale_bias),
        )
        for attr_name, prefill_buffers in quant_specs:
            source = getattr(layer, attr_name, None)
            if source is None or pool_slot >= len(prefill_buffers):
                continue
            components.append((
                attr_name,
                source.data[source_slot],
                prefill_buffers[pool_slot][destination_slot],
            ))
        for name, source, destination in components:
            if source.shape != destination.shape or source.dtype != destination.dtype:
                raise RuntimeError(
                    "Prefill copies require matching expert tensors: "
                    f"layer={layer_idx}, name={name}, "
                    f"source={tuple(source.shape)}/{source.dtype}, "
                    f"destination={tuple(destination.shape)}/{destination.dtype}")
        return components

    def _prefill_peer_components(
        self,
        layer_idx: int,
        pool_slot: int,
        source_slot: int,
        destination_slot: int,
    ) -> list[_PrefillPeerComponent]:
        """Return ordered sources and destinations for one remote expert.

        Weight components always use raw storage because their typed tensors
        may carry an Ascend internal format. Quantization metadata remains
        typed because layouts such as MXFP4 scale are non-contiguous after
        reshape+transpose and require a stride-aware local copy.
        """
        layer = self.moe_layers[layer_idx]
        components: list[_PrefillPeerComponent] = []
        weight_specs = (
            ("w13", self._prefill_w13, self.w13_expert_size_bytes),
            ("w2", self._prefill_w2, self.w2_expert_size_bytes),
        )
        for name, prefill_buffers, nbytes in weight_specs:
            destination_start = destination_slot * nbytes
            components.append(_PrefillPeerComponent(
                name=name,
                source=self._expert_device_storage(
                    layer, source_slot, name),
                destination=prefill_buffers[pool_slot].untyped_storage()[
                    destination_start:destination_start + nbytes],
                nbytes=nbytes,
            ))

        for name, source, destination in self._prefill_device_components(
                layer_idx, pool_slot, source_slot, destination_slot,
                include_weights=False):
            components.append(_PrefillPeerComponent(
                name=name,
                source=source,
                destination=destination,
                nbytes=source.numel() * source.element_size(),
                dtype=source.dtype,
                shape=tuple(source.shape),
                element_size=source.element_size(),
            ))
        return components

    @staticmethod
    def _layout_prefill_peer_components(
        entries: list[tuple[int, _PrefillPeerComponent]],
    ) -> tuple[list[tuple[int, _PrefillPeerComponent, int]], int]:
        """Assign deterministic, dtype-aligned byte offsets to components."""
        layout = []
        offset = 0
        for eid, component in entries:
            alignment = component.element_size
            if alignment <= 0:
                raise ValueError(
                    "Prefill communication component alignment must be "
                    f"positive: name={component.name}, alignment={alignment}")
            offset = ((offset + alignment - 1) // alignment) * alignment
            layout.append((eid, component, offset))
            offset += component.nbytes
        return layout, offset

    @classmethod
    def _prefill_peer_plan_signature(
        cls,
        entries: list[tuple[int, _PrefillPeerComponent]],
    ) -> tuple[int, int, str]:
        """Return a compact deterministic signature for one peer payload."""
        chunks = cls._chunk_prefill_peer_components(
            entries, _EXCLUSIVE_PREFILL_P2P_CHUNK_BYTES)
        description = tuple(
            (
                chunk_idx,
                chunk_bytes,
                tuple(
                    (
                        eid,
                        component.name,
                        offset,
                        component.nbytes,
                        (str(component.dtype)
                         if component.uses_typed_copy else "raw"),
                        component.shape,
                        component.element_size,
                    )
                    for eid, component, offset in layout
                ),
            )
            for chunk_idx, (layout, chunk_bytes) in enumerate(chunks)
        )
        digest = hashlib.sha256(repr(description).encode("utf-8")).hexdigest()
        return (
            sum(chunk_bytes for _, chunk_bytes in chunks),
            sum(len(layout) for layout, _ in chunks),
            digest,
        )

    @classmethod
    def _chunk_prefill_peer_components(
        cls,
        entries: list[tuple[int, _PrefillPeerComponent]],
        max_chunk_bytes: int,
    ) -> list[
        tuple[list[tuple[int, _PrefillPeerComponent, int]], int]]:
        """Split one peer payload into bounded, independently aligned chunks.

        A single component is never split because typed metadata must retain
        its complete shape. Such a component may exceed the soft limit; real
        expert weight and metadata components are much smaller than the
        default limit.
        """
        if max_chunk_bytes <= 0:
            raise ValueError(
                "Prefill P2P chunk size must be positive; "
                f"got {max_chunk_bytes}")
        chunks = []
        current_entries = []
        for entry in entries:
            candidate_entries = current_entries + [entry]
            _, candidate_bytes = cls._layout_prefill_peer_components(
                candidate_entries)
            if current_entries and candidate_bytes > max_chunk_bytes:
                chunks.append(cls._layout_prefill_peer_components(
                    current_entries))
                current_entries = [entry]
            else:
                current_entries = candidate_entries
        if current_entries:
            chunks.append(cls._layout_prefill_peer_components(
                current_entries))
        return chunks

    def _validate_prefill_peer_plans(
        self,
        dist,
        cpu_group,
        layer_idx: int,
        send_components: dict[
            int, list[tuple[int, _PrefillPeerComponent]]],
        recv_components: dict[
            int, list[tuple[int, _PrefillPeerComponent]]],
    ) -> None:
        """Collectively reject asymmetric P2P plans before entering HCCL.

        A send and its matching receive must have the same expert/component
        order, byte offsets, dtypes and shapes. Checking a compact digest on
        the CPU group turns a potential HCCL size mismatch or silent metadata
        corruption into a deterministic initialization error.
        """
        local_plan = {
            "send": {
                peer: self._prefill_peer_plan_signature(entries)
                for peer, entries in sorted(send_components.items())
            },
            "recv": {
                peer: self._prefill_peer_plan_signature(entries)
                for peer, entries in sorted(recv_components.items())
            },
        }
        gathered_plans = [None] * self.ep_size
        dist.all_gather_object(
            gathered_plans, local_plan, group=cpu_group)

        errors = []
        for source_rank, plan in enumerate(gathered_plans):
            if (not isinstance(plan, dict)
                    or not isinstance(plan.get("send"), dict)
                    or not isinstance(plan.get("recv"), dict)):
                errors.append(
                    f"rank {source_rank} published an invalid plan: {plan!r}")
                continue
            for destination_rank, signature in plan["send"].items():
                if not isinstance(destination_rank, int) or not (
                        0 <= destination_rank < self.ep_size):
                    errors.append(
                        f"rank {source_rank} has invalid send peer "
                        f"{destination_rank!r}")
                    continue
                peer_plan = gathered_plans[destination_rank]
                peer_signature = (
                    peer_plan.get("recv", {}).get(source_rank)
                    if isinstance(peer_plan, dict) else None)
                if signature != peer_signature:
                    errors.append(
                        f"send R{source_rank}->R{destination_rank} "
                        f"{signature!r} != recv {peer_signature!r}")
            for source_peer, signature in plan["recv"].items():
                if not isinstance(source_peer, int) or not (
                        0 <= source_peer < self.ep_size):
                    errors.append(
                        f"rank {source_rank} has invalid recv peer "
                        f"{source_peer!r}")
                    continue
                peer_plan = gathered_plans[source_peer]
                peer_signature = (
                    peer_plan.get("send", {}).get(source_rank)
                    if isinstance(peer_plan, dict) else None)
                if signature != peer_signature:
                    errors.append(
                        f"recv R{source_peer}->R{source_rank} "
                        f"{signature!r} != send {peer_signature!r}")
        if errors:
            raise RuntimeError(
                "Cross-rank prefill communication plans do not match: "
                f"layer={layer_idx}; " + "; ".join(errors[:8]))

    @staticmethod
    def _prefill_comm_component_view(
        buffer: torch.Tensor,
        offset: int,
        component: _PrefillPeerComponent,
    ):
        """Return raw or typed local-copy view into a uint8 peer buffer."""
        if component.uses_typed_copy:
            dtype = component.dtype
            shape = component.shape
            if (dtype is None or shape is None
                    or component.nbytes % component.element_size != 0):
                raise RuntimeError(
                    "Invalid typed prefill communication component: "
                    f"name={component.name}, dtype={dtype}, shape={shape}, "
                    f"nbytes={component.nbytes}, "
                    f"element_size={component.element_size}")
            if offset % component.element_size != 0:
                raise RuntimeError(
                    "Misaligned typed prefill communication component: "
                    f"name={component.name}, offset={offset}, "
                    f"element_size={component.element_size}")
            byte_view = buffer.narrow(0, offset, component.nbytes)
            return byte_view.view(dtype).reshape(shape)
        return buffer.untyped_storage()[
            offset:offset + component.nbytes]

    def _get_exclusive_prefill_comm_buffer(
        self,
        direction: str,
        peer_rank: int,
        required_bytes: int,
        device,
    ) -> torch.Tensor:
        """Return a zero-offset contiguous uint8 NPU buffer for one peer."""
        if required_bytes <= 0:
            raise ValueError(
                "Prefill communication buffer size must be positive; "
                f"got {required_bytes}")
        buffers = getattr(self, "_exclusive_prefill_comm_buffers", None)
        if buffers is None:
            buffers = {}
            self._exclusive_prefill_comm_buffers = buffers
        key = (direction, peer_rank)
        buffer = buffers.get(key)
        if (buffer is None or buffer.numel() < required_bytes
                or buffer.device != device):
            # Drop an undersized cached allocation before requesting the
            # larger one so chunk growth does not retain both buffers.
            buffers.pop(key, None)
            buffer = None
            buffer = torch.empty(
                required_bytes, dtype=torch.uint8, device=device)
            buffers[key] = buffer
        communication_view = buffer[:required_bytes]
        if (communication_view.dtype != torch.uint8
                or not communication_view.is_contiguous()
                or communication_view.storage_offset() != 0):
            raise RuntimeError(
                "Invalid cross-rank prefill communication buffer: "
                f"direction={direction}, peer={peer_rank}, "
                f"dtype={communication_view.dtype}, "
                f"contiguous={communication_view.is_contiguous()}, "
                f"storage_offset={communication_view.storage_offset()}")
        return communication_view

    def _get_exclusive_prefill_pyhccl(self, ep_group, device):
        """Collectively create the raw-pointer EP P2P communicator."""
        communicator = getattr(self, "_exclusive_prefill_pyhccl", None)
        if communicator is not None:
            if (not communicator.available or communicator.disabled):
                raise RuntimeError(
                    "Exclusive prefill PyHCCL communicator is closed")
            if communicator.device != device:
                raise RuntimeError(
                    "Exclusive prefill PyHCCL communicator device changed: "
                    f"expected={communicator.device}, actual={device}")
            return communicator

        from torch import distributed as dist
        from vllm_ascend.distributed.device_communicators.pyhccl import (
            PyHcclCommunicator,
        )
        from vllm_ascend.distributed.device_communicators.pyhccl_wrapper import (  # noqa: E501
            HCCLLibrary,
        )

        local_error = None
        try:
            library = HCCLLibrary()
            if not library.supports_batch_send_recv:
                local_error = "HcclBatchSendRecv symbol is missing"
        except BaseException as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        availability = [None] * self.ep_size
        dist.all_gather_object(
            availability, local_error, group=ep_group.cpu_group)
        failures = [
            f"rank={rank} {error}"
            for rank, error in enumerate(availability)
            if error is not None
        ]
        if failures:
            raise RuntimeError(
                "Raw-pointer prefill P2P is unavailable: "
                + "; ".join(failures))

        communicator = PyHcclCommunicator(
            ep_group.cpu_group, device=device)
        if (not communicator.available or communicator.disabled
                or not communicator.hccl.supports_batch_send_recv):
            communicator.close()
            raise RuntimeError(
                "Multi-card exclusive prefill requires the raw-pointer "
                "HcclBatchSendRecv API on every EP rank")
        self._exclusive_prefill_pyhccl = communicator
        logger.info(
            "[PREFILL-D2D] initialized raw-pointer PyHCCL communicator "
            "on %s across %d EP ranks",
            device, self.ep_size)
        return communicator

    def _prefill_load_layer_shard_exclusive_shared(
        self,
        layer_idx: int,
        pool_slot: int,
    ) -> None:
        """Load one static EP shard using CPU H2D and raw-pointer HCCL P2P.

        CPU-owned experts read the compact shared mmap. Same-rank NPU weights
        use the single-card raw-storage D2D representation. Cross-rank NPU
        components are packed into base-format, zero-offset uint8 NPU buffers,
        transferred through ``HcclBatchSendRecv`` on the same load stream,
        then unpacked into the destination prefill pool. Weights use raw
        storage; metadata uses typed local views to preserve non-contiguous
        layouts. This read-only snapshot does not change canonical ownership
        or the decode log2phy mapping.
        """
        from torch import distributed as dist
        from vllm.distributed.parallel_state import get_ep_group

        if not self._shared_h2d_layer_ready(layer_idx):
            raise RuntimeError(
                "Exclusive shared prefill cannot run before the compact "
                f"shared CPU layer is finalized: layer={layer_idx}")
        ep_group = get_ep_group()
        group_ranks = getattr(ep_group, "ranks", None)
        if (getattr(ep_group, "cpu_group", None) is None
                or group_ranks is None
                or len(group_ranks) != self.ep_size):
            raise RuntimeError(
                "Multi-card exclusive prefill requires a complete EP CPU "
                "group; CPU weight fallback is disabled")
        comm_lock = getattr(self, "_exclusive_prefill_comm_lock", None)
        if comm_lock is None:
            comm_lock = threading.Lock()
            self._exclusive_prefill_comm_lock = comm_lock
        layer_lock = self._exclusive_layer_locks[layer_idx]
        with comm_lock, layer_lock:
            device = self._prefill_w13[pool_slot].device
            pyhccl = self._get_exclusive_prefill_pyhccl(
                ep_group, device)

            shard = self.mc_shard_size
            per_rank_slots = (
                self.offload_config.num_device_experts_for_rank(
                    layer_idx, self.ep_size))
            local_tasks: list[H2DCopyTask] = []
            send_components: dict[
                int, list[tuple[int, _PrefillPeerComponent]]] = {}
            recv_components: dict[
                int, list[tuple[int, _PrefillPeerComponent]]] = {}
            remote_experts = 0
            for eid in range(self.num_total_experts):
                destination_rank = eid // shard
                destination_slot = eid % shard
                physical_slot = self._eid_to_npu_slot[layer_idx][eid]
                if physical_slot < 0:
                    # CPU-owned: shared mmap -> pinned staging -> raw NPU
                    # storage.
                    if destination_rank == self.ep_rank:
                        local_tasks.extend(self._build_prefill_h2d_tasks(
                            layer_idx, pool_slot, eid, destination_slot))
                    continue

                source_rank = physical_slot // per_rank_slots
                source_slot = physical_slot % per_rank_slots
                if source_rank == destination_rank:
                    # Same-rank NPU-owned: preserve the single-card raw D2D
                    # path.
                    if source_rank == self.ep_rank:
                        local_tasks.extend(
                            self._build_prefill_local_d2d_tasks(
                                layer_idx, pool_slot, source_slot,
                                destination_slot, eid))
                    continue

                remote_experts += 1
                if self.ep_rank not in (source_rank, destination_rank):
                    continue
                peer_rank = (destination_rank
                             if self.ep_rank == source_rank else source_rank)
                peer_components = (
                    send_components if self.ep_rank == source_rank
                    else recv_components)
                entries = peer_components.setdefault(peer_rank, [])
                entries.extend(
                    (eid, component)
                    for component in self._prefill_peer_components(
                        layer_idx, pool_slot, source_slot,
                        destination_slot)
                )

            self._validate_prefill_peer_plans(
                dist, ep_group.cpu_group, layer_idx,
                send_components, recv_components)

            send_chunks = {
                peer: self._chunk_prefill_peer_components(
                    entries, _EXCLUSIVE_PREFILL_P2P_CHUNK_BYTES)
                for peer, entries in sorted(send_components.items())
            }
            recv_chunks = {
                peer: self._chunk_prefill_peer_components(
                    entries, _EXCLUSIVE_PREFILL_P2P_CHUNK_BYTES)
                for peer, entries in sorted(recv_components.items())
            }
            num_rounds = max(
                [len(chunks) for chunks in send_chunks.values()]
                + [len(chunks) for chunks in recv_chunks.values()]
                + [1])
            total_pack_tasks = sum(
                len(layout)
                for chunks in send_chunks.values()
                for layout, _ in chunks)
            total_unpack_tasks = sum(
                len(layout)
                for chunks in recv_chunks.values()
                for layout, _ in chunks)
            total_p2p_ops = sum(map(len, send_chunks.values()))
            total_p2p_ops += sum(map(len, recv_chunks.values()))
            if self._debug:
                local_h2d_tasks = sum(
                    task.direction == CopyDirection.H2D
                    for task in local_tasks)
                local_d2d_tasks = sum(
                    task.direction == CopyDirection.D2D
                    for task in local_tasks)
                raw_storage_d2d_tasks = sum(
                    task.direction == CopyDirection.D2D
                    and not isinstance(task.source, torch.Tensor)
                    and not isinstance(task.destination, torch.Tensor)
                    for task in local_tasks)
                send_bytes = ",".join(
                    f"{peer}:{sum(size for _, size in chunks)}"
                    for peer, chunks in send_chunks.items()) or "-"
                recv_bytes = ",".join(
                    f"{peer}:{sum(size for _, size in chunks)}"
                    for peer, chunks in recv_chunks.items()) or "-"
                peak_send_chunk = max(
                    (size for chunks in send_chunks.values()
                     for _, size in chunks), default=0)
                peak_recv_chunk = max(
                    (size for chunks in recv_chunks.values()
                     for _, size in chunks), default=0)
                typed_pack_tasks = sum(
                    component.uses_typed_copy
                    for entries in send_components.values()
                    for _, component in entries)
                typed_unpack_tasks = sum(
                    component.uses_typed_copy
                    for entries in recv_components.values()
                    for _, component in entries)
                logger.info(
                    "[PREFILL-D2D-PLAN] layer=%d rank=%d h2d_tasks=%d "
                    "local_d2d_tasks=%d raw_storage_d2d_tasks=%d "
                    "pack_tasks=%d unpack_tasks=%d p2p_ops=%d rounds=%d "
                    "typed_pack_tasks=%d typed_unpack_tasks=%d "
                    "send_bytes_by_peer=%s recv_bytes_by_peer=%s "
                    "chunk_limit=%d peak_send_chunk=%d peak_recv_chunk=%d "
                    "global_remote_experts=%d",
                    layer_idx, self.ep_rank, local_h2d_tasks,
                    local_d2d_tasks, raw_storage_d2d_tasks,
                    total_pack_tasks, total_unpack_tasks, total_p2p_ops,
                    num_rounds, typed_pack_tasks, typed_unpack_tasks,
                    send_bytes, recv_bytes,
                    _EXCLUSIVE_PREFILL_P2P_CHUNK_BYTES,
                    peak_send_chunk, peak_recv_chunk, remote_experts)

            for round_idx in range(num_rounds):
                pack_tasks: list[H2DCopyTask] = []
                unpack_tasks: list[H2DCopyTask] = []
                send_buffers: dict[int, torch.Tensor] = {}
                recv_buffers: dict[int, torch.Tensor] = {}
                for peer_rank, chunks in send_chunks.items():
                    if round_idx >= len(chunks):
                        continue
                    layout, total_bytes = chunks[round_idx]
                    buffer = self._get_exclusive_prefill_comm_buffer(
                        "send", peer_rank, total_bytes, device)
                    send_buffers[peer_rank] = buffer
                    for eid, component, offset in layout:
                        pack_tasks.append(H2DCopyTask(
                            source=component.source,
                            destination=self._prefill_comm_component_view(
                                buffer, offset, component),
                            nbytes=component.nbytes,
                            name=(f"prefill-p2p-pack-{component.name}"
                                  f"[L{layer_idx},E{eid},"
                                  f"R{self.ep_rank}->R{peer_rank}]"),
                            direction=CopyDirection.D2D,
                        ))
                for peer_rank, chunks in recv_chunks.items():
                    if round_idx >= len(chunks):
                        continue
                    layout, total_bytes = chunks[round_idx]
                    buffer = self._get_exclusive_prefill_comm_buffer(
                        "recv", peer_rank, total_bytes, device)
                    recv_buffers[peer_rank] = buffer
                    for eid, component, offset in layout:
                        unpack_tasks.append(H2DCopyTask(
                            source=self._prefill_comm_component_view(
                                buffer, offset, component),
                            destination=component.destination,
                            nbytes=component.nbytes,
                            name=(f"prefill-p2p-unpack-{component.name}"
                                  f"[L{layer_idx},E{eid},"
                                  f"R{peer_rank}->R{self.ep_rank}]"),
                            direction=CopyDirection.D2D,
                        ))

                with torch_npu.npu.stream(self.load_stream):
                    round_local_tasks = (
                        local_tasks if round_idx == 0 else [])
                    self._get_h2d_transport().copy_batch(
                        round_local_tasks + pack_tasks)
                    if send_buffers or recv_buffers:
                        # pack -> HCCL -> unpack use one stream. This avoids
                        # treating ProcessGroup Work.wait() as a cross-stream
                        # dependency for the receive buffer.
                        pyhccl.batch_send_recv(
                            send_buffers, recv_buffers,
                            stream=self.load_stream)
                    self._get_h2d_transport().copy_batch(unpack_tasks)
                    # The next round reuses the same bounded buffers.
                    self._synchronize_h2d()
                del pack_tasks, unpack_tasks
                if send_buffers or recv_buffers:
                    del buffer
                del send_buffers, recv_buffers

            with torch_npu.npu.stream(self.load_stream):
                self._refresh_prefill_fp32_scale(pool_slot, shard)
            self._synchronize_h2d()
            dist.barrier(group=ep_group.cpu_group)
            if self._debug:
                logger.info(
                    "[PREFILL-D2D] layer=%d rank=%d local_tasks=%d "
                    "pack_tasks=%d unpack_tasks=%d p2p_ops=%d "
                    "rounds=%d global_remote_experts=%d",
                    layer_idx, self.ep_rank, len(local_tasks),
                    total_pack_tasks, total_unpack_tasks, total_p2p_ops,
                    num_rounds, remote_experts)

    def _init_prefill_pool_data(self, dev, ntotal: int, ndl: int):
        """Load layer 0 weights into all prefill pool slots.

        Prefill pool tensors are already NZ-cast at this point (done in
        create_prefill_pool). Route the initial full overwrite through the
        configured H2D transport, just like runtime prefill/decode loading.
        """
        del dev
        if self.exclusive_shared_cpu_enabled:
            for slot in range(ndl):
                self._prefill_load_layer_shard_exclusive_shared(0, slot)
            return
        if self.exclusive_sharded_cpu_enabled:
            with torch_npu.npu.stream(self.load_stream):
                for slot in range(ndl):
                    self._load_prefill_shard_from_local_ownership(0, slot)
                self._synchronize_h2d()
            return
        if self.exclusive_dynamic_enabled:
            source_eids = list(range(ntotal))
        else:
            num_source_experts = min(ntotal, len(self.w13_weights_cpu[0]))
            source_eids = [
                (self._shard_base + local_eid
                 if self.offload_config.shard_per_rank else local_eid)
                for local_eid in range(num_source_experts)
            ]
        lock = (self._exclusive_layer_locks[0]
                if self.exclusive_dynamic_enabled else nullcontext())
        with lock:
            with torch_npu.npu.stream(self.load_stream):
                for slot in range(ndl):
                    tasks = []
                    for destination_eid, source_eid in enumerate(source_eids):
                        tasks.extend(self._build_prefill_h2d_tasks(
                            0, slot, source_eid, destination_eid))
                    self._get_h2d_transport().copy_batch(tasks)
                    self._refresh_prefill_fp32_scale(
                        slot, len(source_eids))
                self._synchronize_h2d()

    def _prefill_load_layer(self, layer_idx: int, log2phy: torch.Tensor):
        """Load ALL experts for model layer layer_idx into the prefill pool.

        For W8A8: loads into normal-format scratch, then casts to NZ.
        For unquantized: loads directly into pool tensors via copy_().
        Full-overwrite into pool_slot = layer_idx % ndl.  No slot_owner
        tracking needed — log2phy is set to identity for prefill.
        """
        ndl = self.num_device_layers
        pool_slot = layer_idx % ndl
        ntotal = self.num_total_experts
        is_w8a8 = self._prefill_w13[pool_slot].dtype == torch.int8

        if self._debug:
            logger.info("[PREFILL_LOAD] layer=%d pool_slot=%d ntotal=%d is_w8a8=%s",
                        layer_idx, pool_slot, ntotal, is_w8a8)

        lock = (self._exclusive_layer_locks[layer_idx]
                if self.exclusive_dynamic_enabled else nullcontext())
        with lock:
            with torch_npu.npu.stream(self.load_stream):
                tasks = []
                for eid in range(ntotal):
                    tasks.extend(self._build_prefill_h2d_tasks(
                        layer_idx, pool_slot, eid, eid))
                self._get_h2d_transport().copy_batch(tasks)
                self._refresh_prefill_fp32_scale(pool_slot, ntotal)
                self._synchronize_h2d()

        # NOTE: Do NOT modify the layer's own log2phy here — decode path
        # relies on it staying with 32-expert mapping.  Prefill path in
        # apply() explicitly uses self._prefill_log2phy instead.

    # ------------------------------------------------------------------ #
    #  Multi-card prefill: per-rank EP shard into the prefill pool        #
    # ------------------------------------------------------------------ #
    @property
    def mc_shard_size(self) -> int:
        """Experts per rank in standard EP shard (num_total_experts // ep_size)."""
        return self.num_total_experts // max(1, self.ep_size)

    def _get_shard_expert_map(self) -> torch.Tensor:
        """Standard EP shard expert_map for THIS rank, len = num_total_experts.
        Maps global experts in this rank's shard [base, base+shard) to local
        index [0..shard), everything else to -1. Consumed by the AllGather
        dispatcher (it masks topk_ids to local via expert_map != -1 and uses
        active_expert_range = [rank*nel, rank*nel+nel]).
        """
        if getattr(self, '_mc_shard_expert_map', None) is not None:
            return self._mc_shard_expert_map
        shard = self.mc_shard_size
        base = self.ep_rank * shard
        emap = torch.full((self.num_total_experts,), -1, dtype=torch.int32)
        for i in range(shard):
            emap[base + i] = i
        self._mc_shard_expert_map = emap
        return emap

    def _prefill_load_layer_shard(self, layer_idx: int):
        """Multi-card prefill: load THIS rank's EP shard into the prefill pool.

        Standard EP AllGather has rank r own global experts [r*shard:(r+1)*shard]
        and compute them in LOCAL slots [0:shard]. So we load only the rank's
        shard (not all experts) into pool local slots [0:shard]. The pool buffer
        is sized for num_total_experts; slots [shard:] stay unused this forward.
        Mirrors _prefill_load_layer but sharded + per-rank.
        """
        if not self._prefill_initialized or not self.moe_layers:
            return
        ndl = self.num_device_layers
        pool_slot = layer_idx % ndl
        shard = self.mc_shard_size
        base = self.ep_rank * shard

        if self.exclusive_shared_cpu_enabled:
            self._prefill_load_layer_shard_exclusive_shared(
                layer_idx, pool_slot)
            return

        if self.exclusive_sharded_cpu_enabled:
            with self._exclusive_layer_locks[layer_idx]:
                with torch_npu.npu.stream(self.load_stream):
                    self._load_prefill_shard_from_local_ownership(
                        layer_idx, pool_slot)
                    self._synchronize_h2d()
            return

        if (self.offload_config.h2d_backend == "memfabric"
                or self.torch_shared_cpu_weights_enabled):
            with torch_npu.npu.stream(self.load_stream):
                tasks = []
                for local_i, eid in enumerate(range(base, base + shard)):
                    tasks.extend(self._build_prefill_h2d_tasks(
                        layer_idx, pool_slot, eid, local_i))
                self._get_h2d_transport().copy_batch(tasks)
                self._refresh_prefill_fp32_scale(pool_slot, shard)
                self._synchronize_h2d()
            return

        with torch_npu.npu.stream(self.load_stream):
            for local_i in range(shard):
                eid = base + local_i
                self._prefill_w13[pool_slot].untyped_storage()[local_i * self.w13_expert_size_bytes : (local_i + 1) * self.w13_expert_size_bytes].copy_(
                    self._expert_src_storage(layer_idx, eid, 'w13'))
                self._prefill_w2[pool_slot].untyped_storage()[local_i * self.w2_expert_size_bytes : (local_i + 1) * self.w2_expert_size_bytes].copy_(
                    self._expert_src_storage(layer_idx, eid, 'w2'))
            # quant scales / offsets / scale_bias (w4a8) — shard only
            for scale_name, prefill_list in [("w13_weight_scale", self._prefill_w13_scale),
                                             ("w2_weight_scale", self._prefill_w2_scale)]:
                if pool_slot < len(prefill_list) and scale_name in self.scale_cpu_buffers \
                        and layer_idx < len(self.scale_cpu_buffers[scale_name]):
                    for local_i in range(min(shard, len(self.scale_cpu_buffers[scale_name][layer_idx]))):
                        src = self.scale_cpu_buffers[scale_name][layer_idx][local_i]
                        prefill_list[pool_slot][local_i].copy_(src.reshape(prefill_list[pool_slot][local_i].shape))
            for off_name, prefill_list in [("w13_weight_offset", self._prefill_w13_offset),
                                           ("w2_weight_offset", self._prefill_w2_offset)]:
                if pool_slot < len(prefill_list) and off_name in self.offset_cpu_buffers \
                        and layer_idx < len(self.offset_cpu_buffers[off_name]):
                    for local_i in range(min(shard, len(self.offset_cpu_buffers[off_name][layer_idx]))):
                        src = self.offset_cpu_buffers[off_name][layer_idx][local_i]
                        prefill_list[pool_slot][local_i].copy_(src.reshape(prefill_list[pool_slot][local_i].shape))
            for sb_name, prefill_list in [("w13_scale_bias", self._prefill_w13_scale_bias),
                                          ("w2_scale_bias", self._prefill_w2_scale_bias)]:
                if pool_slot < len(prefill_list) and sb_name in self.scale_bias_cpu_buffers \
                        and layer_idx < len(self.scale_bias_cpu_buffers[sb_name]):
                    for local_i in range(min(shard, len(self.scale_bias_cpu_buffers[sb_name][layer_idx]))):
                        src = self.scale_bias_cpu_buffers[sb_name][layer_idx][local_i]
                        prefill_list[pool_slot][local_i].copy_(src.reshape(prefill_list[pool_slot][local_i].shape))
            # Sync the load stream so the pool data is valid before the GMM reads
            # it (the apply path continues on the default stream after we return;
            # the host-side block here guarantees the copies are done before the
            # next op is queued).
            self._synchronize_h2d()

    def _load_prefill_shard_from_local_ownership(
        self,
        layer_idx: int,
        pool_slot: int,
    ) -> None:
        """Fill this rank's prefill shard without any cross-rank weight read."""
        shard = self.mc_shard_size
        base = self.ep_rank * shard
        tasks = []
        for local_slot, eid in enumerate(range(base, base + shard)):
            if (self._eid_to_cpu_slot[layer_idx][eid] < 0
                    and self._eid_to_npu_slot[layer_idx][eid] < 0):
                raise RuntimeError(
                    "Rank-local exclusive prefill cannot read a remote or "
                    "unowned expert: "
                    f"rank={self.ep_rank}, layer={layer_idx}, expert={eid}")
            tasks.extend(self._build_prefill_h2d_tasks(
                layer_idx, pool_slot, eid, local_slot))
        self._get_h2d_transport().copy_batch(tasks)
        self._refresh_prefill_fp32_scale(pool_slot, shard)

    # ------------------------------------------------------------------ #
    #  Dual-offload test: fill PrefetchOffloader-owned full-size params    #
    # ------------------------------------------------------------------ #
    def _fill_prefetch_full_params(self):
        """Fill w13_weight_full/w2_weight_full/*_scale_full from the
        finalized CPU buffers (same bytes the prefill pool would get).

        Runs inside _finalize_offload, BEFORE offloader.post_init(), so each
        full param's .data is still the prefetch offloader's pinned CPU
        storage; post_init's initial prefetch then H2Ds it into the static
        buffers. Layers not wrapped by the offloader (group not selected,
        MTP drafter) keep their empty NPU tensors and are skipped — apply()
        falls back to the prefill pool for them (no _prefetch_full_ready) —
        UNLESS prefill_module_prefetch_resident is on: their full params are
        cast to the mxfp4 NZ recipe and filled as permanent NPU copies
        (one-time H2D), making the offload-group selection yield truly
        resident layers with zero per-forward copies.
        """
        t0 = time.perf_counter()
        filled = 0
        resident = 0
        resident_bytes = 0
        resident_enabled = (self.offload_config.prefill_module_prefetch
                            and self.offload_config.prefill_module_prefetch_resident)
        for layer_idx, layer in enumerate(self.moe_layers):
            w13_full = getattr(layer, "w13_weight_full", None)
            w2_full = getattr(layer, "w2_weight_full", None)
            if w13_full is None or w2_full is None:
                continue
            on_npu = w13_full.data.device.type != "cpu"
            if on_npu and not resident_enabled:
                # Not wrapped by the prefetch offloader — skip.
                continue
            shard = w13_full.shape[0]
            base = self.ep_rank * self.mc_shard_size
            if on_npu:
                # [dual-offload resident] The empty ND tensor must be cast to
                # the kernel NZ format for this dtype (mirrors the pool's
                # _cast_prefill_pool_format) before the NZ bytes land in it.
                from vllm_ascend.model_executor.offloader.prefetch import (
                    _cast_buffer_to_nz, _kernel_nz_format_for)
                nz_format = _kernel_nz_format_for(w13_full.dtype)
                if nz_format:
                    w13_full.data = _cast_buffer_to_nz(
                        w13_full.data, nz_format)
                    w2_full.data = _cast_buffer_to_nz(
                        w2_full.data, nz_format)
            w13_dst = w13_full.data.untyped_storage()
            w2_dst = w2_full.data.untyped_storage()
            # Weights: raw storage bytes (post-process mxfp4 layout), same
            # recipe as _prefill_load_layer_shard. CPU->CPU for wrapped
            # layers, one-shot H2D for resident ones (same stream as init,
            # so ordering before any forward read is guaranteed).
            for local_i in range(shard):
                geid = base + local_i
                w13_dst[local_i * self.w13_expert_size_bytes:
                        (local_i + 1) * self.w13_expert_size_bytes].copy_(
                            self._expert_src_storage(layer_idx, geid, 'w13'))
                w2_dst[local_i * self.w2_expert_size_bytes:
                       (local_i + 1) * self.w2_expert_size_bytes].copy_(
                           self._expert_src_storage(layer_idx, geid, 'w2'))
            # Scales: per-expert buffers reshaped into the full params,
            # per-format (mxfp4 keeps its encoded uint8 bytes as-is; w8a8's
            # w13 full scale is fp32; w4a8 adds per-channel scale_bias).
            is_mxfp4 = w13_full.dtype == torch.uint8
            scale_names = (["w13_weight_scale", "w2_weight_scale"]
                           + ([] if is_mxfp4 else
                              ["w13_scale_bias", "w2_scale_bias"]))
            cpu_store = {
                "w13_weight_scale": self.scale_cpu_buffers,
                "w2_weight_scale": self.scale_cpu_buffers,
                "w13_scale_bias": self.scale_bias_cpu_buffers,
                "w2_scale_bias": self.scale_bias_cpu_buffers,
            }
            for name in scale_names:
                full = getattr(layer, name + "_full", None)
                if full is None:
                    continue
                cpu_buffers = cpu_store[name].get(name)
                if cpu_buffers is None or layer_idx >= len(cpu_buffers):
                    continue
                for local_i in range(shard):
                    local_eid = self._shard_local(base + local_i)
                    if (local_eid is None
                            or local_eid >= len(cpu_buffers[layer_idx])):
                        continue
                    src = cpu_buffers[layer_idx][local_eid]
                    dst = full.data[local_i]
                    if src.dtype != dst.dtype:
                        src = src.to(dst.dtype)
                    dst.copy_(src.reshape(dst.shape))
            layer._prefetch_full_ready = True
            if on_npu:
                resident += 1
                resident_bytes += (w13_dst.nbytes() + w2_dst.nbytes())
            else:
                filled += 1
        logger.info(
            "[OFFLOAD] prefill_module_prefetch: filled full-size params for "
            "%d/%d MoE layers (%d resident on NPU, %.2f GB) in %.1fs",
            filled + resident, len(self.moe_layers), resident,
            resident_bytes / 1e9, time.perf_counter() - t0)

    # ------------------------------------------------------------------ #
    #  Forward path: page in experts based on topk_ids                    #
    # ------------------------------------------------------------------ #

    def update_weights(self, layer, topk_ids: torch.Tensor,
                        log2phy: torch.Tensor,
                        topk_weights: torch.Tensor | None = None,
                        hidden_states: torch.Tensor | None = None,
                        router_logits: torch.Tensor | None = None,
                        renormalize: bool = False,
                        scoring_func: str = "softmax",
                        e_score_correction_bias: torch.Tensor | None = None,
                        routed_scaling_factor: float = 1.0,
                        is_hash_routed: bool = False) -> int:
        """Incrementally page in needed experts, overwriting unused slots.

        Routes to prefill pool (full-overwrite) when num_tokens exceeds
        offload_threshold, otherwise uses per-expert paging (decode path).

        Args:
            layer: AscendFusedMoE instance.
            topk_ids: [num_tokens, top_k] routed expert indices.
            log2phy: [global_num_experts] CPU tensor, modified in-place.
            topk_weights: Optional routing weights for cache policy.
            hidden_states: Optional [num_tokens, hidden_dim] tensor used
                           for next-layer expert prefetch prediction.

        Returns: number of CPU→NPU copies performed (decode path),
                 0 for prefill path (full-overwrite via pool).
        """
        # Multi-card offload sets routed layers' log2phy=None (standard-EP
        # dispatch) and uses update_weights_multi_card instead. Shared experts
        # (per-card replicated, not dispatched) may still reach here with
        # log2phy=None — bail out to avoid copy_(None). TODO: give shared
        # experts a proper single-card log2phy so they still page in.
        if log2phy is None:
            return 0
        num_tokens = topk_ids.size(0)
        if num_tokens > self.offload_threshold:
            # [dual-offload test] Prefill weights for this layer are served
            # by the PrefetchOffloader's full-size params — skip the pool
            # full-overwrite. Layers without _prefetch_full_ready (drafter,
            # un-wrapped) keep the pool path.
            if (self.offload_config.prefill_module_prefetch
                    and getattr(layer, "_prefetch_full_ready", False)):
                return 0
            # Prefill: layerwise reuse + full-overwrite of all experts
            if (self._prefill_initialized
                    and not self._skip_prefill):
                try:
                    layer_idx = self.moe_layers.index(layer)
                except ValueError:
                    return 0
                self._prefill_load_layer(layer_idx, log2phy)
                return 0
            else:
                # Profile run or pool not ready — bail out gracefully
                return 0

        try:
            layer_idx = self.moe_layers.index(layer)
        except ValueError:
            return 0

        # Wait for prefetch NPU copies to complete before using the weights.
        # Use stream wait (graphable) instead of host synchronize.
        with self._prefetch_state_lock:
            npu_event = self._prefetch_layer_npu_event.pop(layer_idx, None)
        if npu_event is not None:
            torch_npu.npu.current_stream().wait_event(npu_event)

        topk_ids_h = self.topk_ids_h[:num_tokens]
        do_substitution = (
            self.offload_config.expert_substitution_enabled
            and not is_hash_routed
            and router_logits is not None
            and router_logits.shape[-1] == self.num_total_experts
        )
        if (self.offload_config.expert_substitution_enabled
                and not is_hash_routed and router_logits is not None
                and router_logits.shape[-1] != self.num_total_experts):
            logger.warning_once(
                "[SUBST] router_logits width %d != num_total_experts %d; "
                "expert substitution is disabled for this layer",
                router_logits.shape[-1], self.num_total_experts)

        topk_weights_h = None
        if (topk_weights is not None and self.cache_policy is not None
                and self.offload_config.cache_router_weight != 0):
            topk_weights_h = self.topk_weights_h[:num_tokens]
            topk_weights_h.copy_(topk_weights.to(dtype=torch.float32), non_blocking=_EXTRA_CTX.capturing)
        router_logits_h = None
        correction_bias_h = None
        if do_substitution:
            router_logits_h = self.router_logits_h[:num_tokens]
            router_logits_h.copy_(router_logits.to(dtype=torch.float32),
                                  non_blocking=_EXTRA_CTX.capturing)
            if e_score_correction_bias is not None:
                correction_bias_h = self.e_score_correction_bias_h
                correction_bias_h.copy_(
                    e_score_correction_bias.to(dtype=torch.float32),
                    non_blocking=_EXTRA_CTX.capturing)
        log2phy_h = self.log2phy_h
        log2phy_np = self.log2phy_np
        topk_ids_h.copy_(topk_ids, non_blocking=_EXTRA_CTX.capturing)
        log2phy_h.copy_(log2phy, non_blocking=_EXTRA_CTX.capturing)

        current_compute_stream = torch_npu.npu.current_stream()
        subscribed_compute_streams = get_subscribed_compute_streams()
        if current_compute_stream not in subscribed_compute_streams:
            torch_npu.npu._subscribe_report(current_compute_stream)
            subscribed_compute_streams.add(current_compute_stream)
        self._is_prefetch = False
        args = (
            topk_ids_h,
            log2phy_np,
            layer,
            layer_idx,
            topk_weights_h,
            self._is_prefetch,
            do_substitution,
            router_logits_h,
            scoring_func,
            correction_bias_h,
        )
        if _EXTRA_CTX.capturing:
            torch_npu.npu._launch_host_func(
                current_compute_stream,
                self._update_weights,
                args,
            )
        else:
            self._update_weights(args)

        if do_substitution:
            topk_ids[:, :self.topk].copy_(topk_ids_h[:, :self.topk],
                                          non_blocking=True)
        log2phy.copy_(log2phy_h, non_blocking=_EXTRA_CTX.capturing)

    def _mc_handle_prefill_regime(self, layer_idx) -> bool:
        """Multi-card PREFILL (non-MC2 comm): load this rank's EP shard into the
        prefill pool. Returns True when handled so the caller returns early.

        We load even during profile_run: the decode buffer is too small for the
        EP shard, so multi-card prefill MUST use the pool, and real weights
        avoid GMM errors on garbage scales (single-card can skip during profile
        because its AllGather reuses decode weights; multi-card All2All cannot).
        """
        from vllm_ascend.ascend_forward_context import MoECommType
        if _EXTRA_CTX.moe_comm_type == MoECommType.MC2:
            return False
        if self._prefill_initialized:
            # [dual-offload test] Layers whose full-size params were filled
            # are served by the PrefetchOffloader in prefill — skip the pool
            # shard load. Still return True (prefill regime handled).
            if (self.offload_config.prefill_module_prefetch
                    and getattr(self.moe_layers[layer_idx],
                                "_prefetch_full_ready", False)):
                return True
            self._prefill_load_layer_shard(layer_idx)
            if self._debug and logger.isEnabledFor(logging.DEBUG):
                base = self.ep_rank * self.mc_shard_size
                logger.debug(
                    "[MC_OBS] rank=%s L=%s PREFILL: loaded EP shard "
                    "experts[%d..%d] (%d experts) into pool on rank%d "
                    "(static shard, reloaded each prefill forward)",
                    self.ep_rank, layer_idx, base, base + self.mc_shard_size - 1,
                    self.mc_shard_size, self.ep_rank)
        return True

    def _log_mc_debug_event(self, event: str, context=None, **details) -> None:
        """Emit one parseable CPU-only multi-card diagnostic record."""
        if not self._debug:
            return
        context = context or {}
        details_text = " ".join(
            f"{key}={value}" for key, value in details.items())
        logger.info(
            "[MC_DEBUG] event=%s rank=%s layer=%s layer_call=%s "
            "callback_seq=%s source=%s prefetch=%s pid=%s thread=%s "
            "ts_ns=%s %s",
            event,
            self.ep_rank,
            context.get("layer_idx", "-"),
            context.get("layer_call", "-"),
            context.get("callback_seq", "-"),
            context.get("source", "-"),
            context.get("is_prefetch", False),
            os.getpid(),
            threading.get_ident(),
            time.time_ns(),
            details_text,
        )

    def log_exclusive_sharded_numeric(self, layer, tensor: object,
                                      stage: str) -> None:
        """Report the first finite result and every non-finite local result.

        This diagnostic is deliberately limited to rank-local exclusive mode.
        ``item()`` synchronizes the NPU, which is acceptable under
        ``moe_offload_debug`` while validating an eager path but is forbidden
        while ACL Graph capture is active. Diagnostics must not abort model
        startup if a caller accidentally passes a result wrapper.
        """
        if not self._debug or not self.exclusive_sharded_cpu_enabled:
            return
        try:
            layer_idx = self.moe_layers.index(layer)
        except ValueError:
            return
        if not isinstance(tensor, torch.Tensor):
            logger.warning(
                "[EXCLUSIVE-SHARDED-NUMERIC] rank=%d/%d layer=%d stage=%s "
                "input_type=%s status=skip reason=not_tensor",
                self.ep_rank, self.ep_size, layer_idx, stage,
                type(tensor).__name__)
            return
        if bool(getattr(_EXTRA_CTX, "capturing", False)):
            logger.info_once(
                "[EXCLUSIVE-SHARDED-NUMERIC] skip NPU value inspection "
                "during ACL Graph capture; other offload debug diagnostics "
                "remain enabled")
            return
        nonfinite = int((~torch.isfinite(tensor)).sum().item())
        comm_type = getattr(_EXTRA_CTX.moe_comm_type, "name",
                            _EXTRA_CTX.moe_comm_type)
        if nonfinite:
            logger.warning(
                "[EXCLUSIVE-SHARDED-NUMERIC] rank=%d/%d layer=%d stage=%s "
                "comm=%s shape=%s dtype=%s total=%d nonfinite=%d status=fail",
                self.ep_rank, self.ep_size, layer_idx, stage, comm_type,
                tuple(tensor.shape), tensor.dtype, tensor.numel(), nonfinite)
            return
        seen = getattr(self, "_exclusive_sharded_numeric_seen", None)
        if seen is None:
            seen = set()
            self._exclusive_sharded_numeric_seen = seen
        key = (layer_idx, stage, str(comm_type))
        if key in seen:
            return
        seen.add(key)
        logger.info(
            "[EXCLUSIVE-SHARDED-NUMERIC] rank=%d/%d layer=%d stage=%s "
            "comm=%s shape=%s dtype=%s total=%d nonfinite=0 status=ok",
            self.ep_rank, self.ep_size, layer_idx, stage, comm_type,
            tuple(tensor.shape), tensor.dtype, tensor.numel())

    def _log_mc_debug_schedule(self, layer_idx: int,
                               is_prefetch: bool) -> None:
        if not self._debug:
            return
        with self._mc_debug_lock:
            self._mc_debug_schedule_seq += 1
            schedule_seq = self._mc_debug_schedule_seq
        self._log_mc_debug_event(
            "CB_SCHEDULE",
            {
                "layer_idx": layer_idx,
                "source": "graph_callback",
                "is_prefetch": is_prefetch,
            },
            schedule_seq=schedule_seq,
        )

    def _begin_mc_debug_callback(self, layer_idx: int, is_prefetch: bool,
                                 from_graph_callback: bool):
        if not self._debug:
            return None
        with self._mc_debug_lock:
            self._mc_debug_callback_seq += 1
            callback_seq = self._mc_debug_callback_seq
            layer_key = (layer_idx, is_prefetch)
            layer_call = self._mc_debug_layer_calls.get(layer_key, 0) + 1
            self._mc_debug_layer_calls[layer_key] = layer_call
            self._mc_debug_active_callbacks += 1
            active_callbacks = self._mc_debug_active_callbacks
        context = {
            "layer_idx": layer_idx,
            "layer_call": layer_call,
            "callback_seq": callback_seq,
            "source": ("graph_callback" if from_graph_callback
                       else "eager_inline"),
            "is_prefetch": is_prefetch,
            "start_ns": time.perf_counter_ns(),
        }
        self._log_mc_debug_event(
            "CB_ENTER", context, active_callbacks=active_callbacks)
        return context

    def _end_mc_debug_callback(self, context, status: str) -> None:
        if context is None:
            return
        with self._mc_debug_lock:
            self._mc_debug_active_callbacks = max(
                0, self._mc_debug_active_callbacks - 1)
            active_callbacks = self._mc_debug_active_callbacks
        elapsed_us = (time.perf_counter_ns() - context["start_ns"]) // 1000
        self._log_mc_debug_event(
            "CB_EXIT",
            context,
            status=status,
            active_callbacks=active_callbacks,
            elapsed_us=elapsed_us,
        )

    def _gather_cpu_with_mc_debug(self, local_values, cpu_group, kind: str,
                                  context):
        """Wrap one Gloo all-reduce with enter/exit sequence diagnostics."""
        from vllm_ascend.expert_offload.multi_card_planner import (
            gather_global_counts_cpu)

        if context is None:
            return gather_global_counts_cpu(local_values, cpu_group)
        with self._mc_debug_lock:
            self._mc_debug_collective_seq += 1
            collective_seq = self._mc_debug_collective_seq
        start_ns = time.perf_counter_ns()
        group_name = getattr(cpu_group, "group_name", "none")
        group_id = hex(id(cpu_group)) if cpu_group is not None else "none"
        self._log_mc_debug_event(
            "GLOO_ENTER",
            context,
            kind=kind,
            collective_seq=collective_seq,
            group_name=group_name,
            local_group_id=group_id,
            dtype=local_values.dtype,
            numel=local_values.numel(),
        )
        try:
            global_values = gather_global_counts_cpu(local_values, cpu_group)
        except BaseException as exc:
            self._log_mc_debug_event(
                "GLOO_ERROR",
                context,
                kind=kind,
                collective_seq=collective_seq,
                error_type=type(exc).__name__,
            )
            raise
        elapsed_us = (time.perf_counter_ns() - start_ns) // 1000
        self._log_mc_debug_event(
            "GLOO_EXIT",
            context,
            kind=kind,
            collective_seq=collective_seq,
            elapsed_us=elapsed_us,
        )
        return global_values

    def update_weights_multi_card(self, layer, topk_ids, log2phy,
                                  topk_weights=None, hidden_states=None,
                                  mc2_mask=None,
                                  router_logits=None,
                                  renormalize=False,
                                  scoring_func="softmax",
                                  e_score_correction_bias=None,
                                  routed_scaling_factor=1.0,
                                  is_hash_routed=False):
        """Multi-card EP offload: planner decides global placement; this rank
        H2D-loads only its assigned experts and writes the placement into
        ``log2phy`` (which the MC2 dispatcher then consumes).

        MVP: full H2D of this rank's assigned experts every layer. No
        skip-if-resident, no LRC victim selection, no hot pool yet (those are
        later stages). Determinism comes from the planner (decision 8): every
        rank feeds the same all-reduced counts and gets the same placement.
        """
        try:
            layer_idx = self.moe_layers.index(layer)
        except ValueError:
            return
        # Wait for this layer's prefetch (if any) to finish H2D before reading
        # the device slots — mirror the single-card update_weights stream-join
        # (graphable: stream wait_event, not host sync). Without it the reactive
        # GMM could read a slot before the prefetch's load_stream H2D lands.
        with self._prefetch_state_lock:
            npu_event = self._prefetch_layer_npu_event.pop(layer_idx, None)
        if npu_event is not None:
            torch_npu.npu.current_stream().wait_event(npu_event)
        num_tokens = topk_ids.size(0)

        # NOTE: the decode profile dummy must use the real dynamic placement
        # (NOT a spread shortcut) — spread maps dummy topk all to rank0 and
        # deadlocks MC2. Debug observability + the overflow-spread fallback now
        # live in _update_weights_multi_card (graph-safe: they read the pinned
        # CPU topk_ids_h, not the live NPU tensor).

        # PREFILL regime: the MC2 dispatch kernel caps at 512 tokens, so prefill
        # uses ALLTOALL + a per-rank EP shard loaded into the prefill pool
        # (selected by select_moe_comm_method -> ALLTOALL for multi-card large
        # batches). Drive off the comm TYPE (MC2=decode, else prefill) — the
        # single source of truth — so this stays in lockstep with apply().
        if self._mc_handle_prefill_regime(layer_idx):
            return
        # ---- DECODE (MC2) branch: graph-aware (mirror single-card) ----
        # Dynamic placement varies per step (router-driven) and is incompatible
        # with cudagraph's fixed op sequence. Mirror single-card update_weights:
        # D2H router outputs into pinned CPU buffers, run planning+H2D as a host
        # callback (_launch_host_func, re-executed every replay with the current
        # topk) or inline (eager), H2D log2phy back. The host callback's
        # load_stream.synchronize() gates the compute stream until H2D is done.
        # The cross-rank expert-count all_reduce uses gloo cpu_group
        # (get_ep_group().cpu_group), while HCCL all_reduce cannot be a captured
        # graph op and the planner needs the counts on host. Every EP rank must
        # still enter the Gloo collectives in exactly the same order; MC_DEBUG
        # traces that ordering across graph host callbacks.
        per_rank_slots = self.offload_config.num_device_experts_for_rank(
            layer_idx, self.ep_size)
        topk_ids_h = self.topk_ids_h[:num_tokens]
        log2phy_h = self.log2phy_h
        topk_ids_h.copy_(topk_ids, non_blocking=_EXTRA_CTX.capturing)
        log2phy_h.copy_(log2phy, non_blocking=_EXTRA_CTX.capturing)
        do_substitution = (
            self.offload_config.expert_substitution_enabled
            and not is_hash_routed
            and router_logits is not None
            and router_logits.shape[-1] == self.num_total_experts
        )
        if (self.offload_config.expert_substitution_enabled
                and not is_hash_routed and router_logits is not None
                and router_logits.shape[-1] != self.num_total_experts):
            logger.warning_once(
                "[SUBST-MC] router_logits width %d != num_total_experts %d; "
                "expert substitution is disabled for this layer",
                router_logits.shape[-1], self.num_total_experts)
        router_logits_h = None
        correction_bias_h = None
        if do_substitution:
            router_logits_h = self.router_logits_h[:num_tokens]
            router_logits_h.copy_(
                router_logits.to(dtype=torch.float32),
                non_blocking=_EXTRA_CTX.capturing)
            if e_score_correction_bias is not None:
                correction_bias_h = self.e_score_correction_bias_h
                correction_bias_h.copy_(
                    e_score_correction_bias.to(dtype=torch.float32),
                    non_blocking=_EXTRA_CTX.capturing)
        # Mirror the per-rank active-token mask to pinned CPU on the same
        # stream as topk_ids_h, so the host callback (graph replay) reads it
        # after the copy lands — same ordering contract as topk_ids_h. None
        # means all-active (e.g. non-uniform global_bs path): no filtering,
        # fully backward compatible.
        if mc2_mask is not None:
            mc2_mask_h = self.mc2_mask_h[:num_tokens]
            # Cast bool->int32 on the NPU first: a direct bool D2H on the
            # captured stream forces a sync ("stream is captured", rtMemcpy
            # 107027) because Ascend has no async bool memcpy path. int32 D2H
            # is the same async path topk_ids_h.copy_ already uses, so it
            # records cleanly into the graph.
            mc2_mask_h.copy_(mc2_mask.to(torch.int32),
                             non_blocking=_EXTRA_CTX.capturing)
        else:
            mc2_mask_h = None
        current_compute_stream = torch_npu.npu.current_stream()
        subscribed = get_subscribed_compute_streams()
        if current_compute_stream not in subscribed:
            torch_npu.npu._subscribe_report(current_compute_stream)
            subscribed.add(current_compute_stream)
        topk_weights_h = None
        if (topk_weights is not None and self.cache_policy is not None
                and self.offload_config.cache_router_weight != 0):
            topk_weights_h = self.topk_weights_h[:num_tokens]
            topk_weights_h.copy_(topk_weights.to(dtype=torch.float32),
                                 non_blocking=_EXTRA_CTX.capturing)
        from_graph_callback = _EXTRA_CTX.capturing
        args = (
            topk_ids_h, log2phy_h, layer, layer_idx, per_rank_slots, False,
            mc2_mask_h, do_substitution, router_logits_h, scoring_func,
            correction_bias_h, topk_weights_h,
        )
        if self._debug:
            args += (from_graph_callback,)
        if from_graph_callback:
            self._log_mc_debug_schedule(layer_idx, is_prefetch=False)
            torch_npu.npu._launch_host_func(
                current_compute_stream, self._update_weights_multi_card, args)
        else:
            self._update_weights_multi_card(args)
        if do_substitution:
            topk_ids[:, :self.topk].copy_(
                topk_ids_h[:, :self.topk], non_blocking=True)
        # Copy the (host-func-mutated) log2phy_h back to the NPU tensor so the
        # MC2 dispatcher reads the fresh placement.
        log2phy.copy_(log2phy_h, non_blocking=_EXTRA_CTX.capturing)

    def _expert_src_storage(self, layer_idx, eid, which='w13'):
        """Return expert eid's bytes as UntypedStorage for H2D **read**.

        shard-per-rank: eid is GLOBAL; remap to this rank's local shard slot.
        Otherwise: the expert tensor's own (already pinned) storage, global eid.
        """
        cpu_buf = getattr(self, f'{which}_weights_cpu')[layer_idx]
        if self.exclusive_dynamic_enabled:
            local_eid = self._cpu_local(layer_idx, eid)
            if local_eid is None:
                raise RuntimeError(
                    "Requested CPU source for an NPU-owned expert: "
                    f"layer={layer_idx}, expert={eid}, weight={which}")
            if self.exclusive_shared_cpu_enabled:
                return self._torch_shared_cpu_weight_storage(
                    layer_idx, eid, which)
            return cpu_buf[local_eid].untyped_storage()
        if self.torch_shared_cpu_weights_enabled:
            return self._torch_shared_cpu_weight_storage(
                layer_idx, eid, which)
        if self.offload_config.shard_per_rank:
            local_eid = self._shard_local(eid)
            if local_eid is not None:
                return cpu_buf[local_eid].untyped_storage()
            return self._shared_h2d_source(layer_idx, eid, which)
        return cpu_buf[eid].untyped_storage()

    def _expert_device_storage(self, layer, slot: int, which: str):
        """Return one decode slot's raw device bytes without changing layout."""
        size = (self.w13_expert_size_bytes if which == "w13"
                else self.w2_expert_size_bytes)
        start = slot * size
        return _expert_weight(
            layer, f"{which}_weight").data.untyped_storage()[
                start:start + size]

    def _expert_dst_storage(self, layer_idx, eid, which='w13'):
        """Return expert eid's storage for fill **write**.

        shard-per-rank: eid is GLOBAL; remap to this rank's local shard slot.
        Otherwise: the expert tensor's own storage, global eid.
        """
        cpu_buf = getattr(self, f'{which}_weights_cpu')[layer_idx]
        if self.exclusive_dynamic_enabled:
            local_eid = self._cpu_local(layer_idx, eid)
            if local_eid is None:
                raise RuntimeError(
                    "Requested CPU destination for an NPU-owned expert: "
                    f"layer={layer_idx}, expert={eid}, weight={which}")
            if self.exclusive_shared_cpu_enabled:
                return self._torch_shared_cpu_weight_storage(
                    layer_idx, eid, which)
            return cpu_buf[local_eid].untyped_storage()
        if self.torch_shared_cpu_weights_enabled:
            return self._torch_shared_cpu_weight_storage(
                layer_idx, eid, which)
        if self.offload_config.shard_per_rank:
            return cpu_buf[eid - self._shard_base].untyped_storage()
        return cpu_buf[eid].untyped_storage()

    def _torch_shared_cpu_weight_storage(self, layer_idx, eid, which):
        tensor = self._torch_shared_cpu_buffers.get((layer_idx, which))
        if tensor is None:
            raise RuntimeError(
                "Missing Torch shared CPU weight buffer: "
                f"layer={layer_idx}, expert={eid}, name={which}")
        per_expert = (self.w13_expert_size_bytes if which == "w13"
                      else self.w2_expert_size_bytes)
        shared_slot = eid
        if self.exclusive_shared_cpu_enabled:
            shared_slot = self._cpu_local(layer_idx, eid)
            if shared_slot is None:
                raise RuntimeError(
                    "Requested shared CPU source for an NPU-owned expert: "
                    f"layer={layer_idx}, expert={eid}, weight={which}")
        start = shared_slot * per_expert
        return tensor.untyped_storage()[start:start + per_expert]

    def _torch_shared_cpu_attr_source(self, layer_idx, eid, name):
        tensor = self._torch_shared_cpu_buffers.get((layer_idx, name))
        if tensor is None:
            raise RuntimeError(
                "Missing Torch shared CPU quantization buffer: "
                f"layer={layer_idx}, expert={eid}, name={name}")
        shared_slot = eid
        if self.exclusive_shared_cpu_enabled:
            shared_slot = self._cpu_local(layer_idx, eid)
            if shared_slot is None:
                raise RuntimeError(
                    "Requested shared CPU quantization source for an "
                    f"NPU-owned expert: layer={layer_idx}, expert={eid}, "
                    f"name={name}")
        return tensor[shared_slot]

    def _shared_h2d_source(self, layer_idx, eid, name):
        if self.torch_shared_cpu_weights_enabled:
            if name in ("w13", "w2"):
                return self._torch_shared_cpu_weight_storage(
                    layer_idx, eid, name)
            return self._torch_shared_cpu_attr_source(
                layer_idx, eid, name)
        pointer = self._shared_h2d_sources.get((layer_idx, eid, name))
        if pointer is None:
            raise RuntimeError(
                "Missing MemFabric SHARED source pointer: "
                f"layer={layer_idx}, expert={eid}, name={name}")
        return HostPointerSource(pointer)

    def _shared_h2d_layer_ready(self, layer_idx: int) -> bool:
        if self.torch_shared_cpu_weights_enabled:
            return (
                getattr(self, '_torch_shared_cpu_sources_ready', False)
                and layer_idx in self._torch_shared_cpu_ready_layers
                and (layer_idx, "w13") in self._torch_shared_cpu_buffers
                and (layer_idx, "w2") in self._torch_shared_cpu_buffers
            )
        if not getattr(self, '_shared_h2d_sources_ready', False):
            return False
        shared_sources = getattr(self, '_shared_h2d_sources', {})
        return all(
            (layer_idx, eid, name) in shared_sources
            for eid in range(self.num_total_experts)
            for name in ("w13", "w2"))

    def _publish_shared_h2d_sources(self) -> None:
        """Make all same-node shared sources visible after conversion."""
        if self.torch_shared_cpu_weights_enabled:
            from torch import distributed as dist
            from vllm.distributed.parallel_state import get_ep_group
            dist.barrier(group=get_ep_group().cpu_group)
            self._torch_shared_cpu_ready_layers.update(
                range(len(self.w13_weights_cpu)))
            self._torch_shared_cpu_sources_ready = True
            if not all(self._shared_h2d_layer_ready(layer_idx)
                       for layer_idx in range(len(self.w13_weights_cpu))):
                raise RuntimeError("Incomplete Torch shared CPU source table")
            self._log_torch_shared_cpu_pss()
            logger.info(
                "[EXPERT-OFFLOAD-H2D] Torch shared CPU sources ready: "
                "layers=%d experts=%d ranks=%d",
                len(self.w13_weights_cpu), self.num_total_experts,
                self.ep_size)
            return

        # MemFabric SHARED publishes pointer-valued peer shard sources.
        transport = self._get_h2d_transport()
        if not getattr(transport, "supports_remote_sources", False):
            return

        local_sources = {}
        for layer_idx in range(len(self.w13_weights_cpu)):
            for local_eid, tensor in enumerate(
                    self.w13_weights_cpu[layer_idx]):
                eid = self._shard_base + local_eid
                local_sources[(layer_idx, eid, "w13")] = tensor.data_ptr()
                local_sources[(layer_idx, eid, "w2")] = (
                    self.w2_weights_cpu[layer_idx][local_eid].data_ptr())
        for buffer_dict in (self.scale_cpu_buffers,
                            self.offset_cpu_buffers,
                            self.scale_bias_cpu_buffers):
            for name, layer_buffers in buffer_dict.items():
                for layer_idx, expert_buffers in enumerate(layer_buffers):
                    for local_eid, tensor in enumerate(expert_buffers):
                        eid = self._shard_base + local_eid
                        local_sources[(layer_idx, eid, name)] = tensor.data_ptr()

        from torch import distributed as dist
        from vllm.distributed.parallel_state import get_ep_group
        gathered_sources = [None] * self.ep_size
        dist.all_gather_object(
            gathered_sources, local_sources,
            group=get_ep_group().cpu_group)
        shared_sources = {}
        for rank_sources in gathered_sources:
            overlap = shared_sources.keys() & rank_sources.keys()
            if overlap:
                raise RuntimeError(
                    "Duplicate MemFabric SHARED expert sources: "
                    f"{list(overlap)[:4]}")
            shared_sources.update(rank_sources)
        self._shared_h2d_sources = shared_sources
        self._shared_h2d_sources_ready = True
        if not all(self._shared_h2d_layer_ready(layer_idx)
                   for layer_idx in range(len(self.w13_weights_cpu))):
            raise RuntimeError("Incomplete MemFabric SHARED source table")
        logger.info(
            "[EXPERT-OFFLOAD-H2D] published %d SHARED source pointers "
            "across %d EP ranks", len(shared_sources), self.ep_size)

    def _initialize_exclusive_shared_runtime_state(self) -> None:
        """Publish initial global NPU ownership to decode runtime state."""
        per_rank_slots_by_layer = [
            self.offload_config.num_device_experts_for_rank(
                layer_idx, self.ep_size)
            for layer_idx in range(len(self.moe_layers))
        ]
        for layer_idx, layer in enumerate(self.moe_layers):
            placement = torch.full(
                (self.num_total_experts,), -1, dtype=torch.int32)
            for physical_slot, eid in enumerate(
                    self._npu_slot_to_eid[layer_idx]):
                placement[eid] = physical_slot
            per_rank_slots = per_rank_slots_by_layer[layer_idx]
            base = self.ep_rank * per_rank_slots
            local_eids = self._npu_slot_to_eid[layer_idx][
                base:base + per_rank_slots]
            self._mc_prev_log2phy[layer_idx] = placement.clone()
            self._mc_resident[layer_idx] = {
                slot: int(eid) for slot, eid in enumerate(local_eids)
            }
            layer.log2phy.copy_(placement)
            self._validate_exclusive_ownership(layer_idx)
        logger.info(
            "[EXCLUSIVE-SHARED] initialized global canonical ownership: "
            "layers=%d global_npu_experts=%s compact_cpu_experts=%s "
            "ranks=%d",
            len(self.moe_layers),
            [len(slots) for slots in self._npu_slot_to_eid],
            [len(slots) for slots in self._cpu_slot_to_eid],
            self.ep_size)

    def _initialize_exclusive_sharded_runtime_state(self) -> None:
        """Publish deterministic rank-local ownership to decode state."""
        for layer_idx, layer in enumerate(self.moe_layers):
            per_rank_slots = self.offload_config.num_device_experts_for_rank(
                layer_idx, self.ep_size)
            placement = torch.full(
                (self.num_total_experts,), -1, dtype=torch.int32)
            for rank in range(self.ep_size):
                rank_eids = (
                    self.offload_config.initial_device_experts_for_rank(
                        layer_idx, self.num_total_experts,
                        self.ep_size, rank))
                for local_slot, eid in enumerate(rank_eids):
                    placement[eid] = rank * per_rank_slots + local_slot

            expected_local = (
                self.offload_config.initial_device_experts_for_rank(
                    layer_idx, self.num_total_experts,
                    self.ep_size, self.ep_rank))
            if self._npu_slot_to_eid[layer_idx] != expected_local:
                raise RuntimeError(
                    "Rank-local exclusive NPU initialization mismatch: "
                    f"rank={self.ep_rank}, layer={layer_idx}, "
                    f"expected={expected_local}, "
                    f"actual={self._npu_slot_to_eid[layer_idx]}")
            self._mc_prev_log2phy[layer_idx] = placement.clone()
            self._mc_resident[layer_idx] = {
                slot: int(eid)
                for slot, eid in enumerate(expected_local)
            }
            layer.log2phy.copy_(placement)
            self._validate_exclusive_ownership(layer_idx)
        logger.info(
            "[EXCLUSIVE-SHARDED] initialized rank-local canonical ownership: "
            "rank=%d/%d layers=%d shard=[%d,%d) "
            "cpu_experts_per_layer=%s npu_experts_per_layer=%s "
            "cross_rank_weight_transfer=False",
            self.ep_rank, self.ep_size, len(self.moe_layers),
            self._shard_base, self._shard_base + self._shard_size,
            [len(slots) for slots in self._cpu_slot_to_eid],
            [len(slots) for slots in self._npu_slot_to_eid])

    def _log_exclusive_shared_canonical_memory(self) -> None:
        """Audit that compact CPU plus all decode NPU slots equal one model."""
        if not self._debug or not self.exclusive_shared_cpu_enabled:
            return
        from torch import distributed as dist
        from vllm.distributed.parallel_state import get_ep_group

        canonical_attr_names = (
            "w13_weight_scale",
            "w2_weight_scale",
            "w13_weight_offset",
            "w2_weight_offset",
            "w13_scale_bias",
            "w2_scale_bias",
        )
        local_device_tensors = []
        for layer in self.moe_layers:
            local_device_tensors.extend((
                _expert_weight(layer, "w13_weight").data,
                _expert_weight(layer, "w2_weight").data,
            ))
            local_device_tensors.extend(
                getattr(layer, name).data
                for name in canonical_attr_names
                if hasattr(layer, name)
            )
        local_npu_bytes = self._cpu_tensor_storage_bytes(
            local_device_tensors)
        shared_cpu_bytes = self._cpu_tensor_storage_bytes(
            self._torch_shared_cpu_buffers)
        expected_full_bytes = 0
        for (layer_idx, _), tensor in self._torch_shared_cpu_buffers.items():
            cpu_experts = len(self._cpu_slot_to_eid[layer_idx])
            expected_full_bytes += (
                tensor.untyped_storage().nbytes()
                // cpu_experts * self.num_total_experts)

        snapshots = [None] * self.ep_size
        dist.all_gather_object(
            snapshots,
            (shared_cpu_bytes, local_npu_bytes),
            group=get_ep_group().cpu_group,
        )
        if self.ep_rank != 0:
            return
        cpu_sizes = [snapshot[0] for snapshot in snapshots]
        global_npu_bytes = sum(snapshot[1] for snapshot in snapshots)
        actual_full_bytes = shared_cpu_bytes + global_npu_bytes
        ownership_ok = all(
            len(self._cpu_slot_to_eid[layer_idx])
            + len(self._npu_slot_to_eid[layer_idx])
            == self.num_total_experts
            for layer_idx in range(len(self.moe_layers))
        )
        result = (
            "PASS_ONE_CANONICAL_MODEL"
            if (ownership_ok and len(set(cpu_sizes)) == 1
                and actual_full_bytes == expected_full_bytes)
            else "FAIL_CANONICAL_MEMORY_MISMATCH"
        )
        mib = 1024 ** 2
        logger.info(
            "[EXCLUSIVE-SHARED-AUDIT] ranks=%d layers=%d "
            "compact_shared_cpu_mib=%.1f global_decode_npu_mib=%.1f "
            "canonical_total_mib=%.1f expected_one_model_mib=%.1f "
            "cpu_experts_per_layer=%s global_npu_experts_per_layer=%s "
            "result=%s",
            self.ep_size, len(self.moe_layers),
            shared_cpu_bytes / mib, global_npu_bytes / mib,
            actual_full_bytes / mib, expected_full_bytes / mib,
            [len(slots) for slots in self._cpu_slot_to_eid],
            [len(slots) for slots in self._npu_slot_to_eid],
            result)

    def _build_quant_attr_h2d_tasks(self, layer, layer_idx, eid,
                                    slot) -> list[H2DCopyTask]:
        """Build scale/offset/scale_bias H2D tasks for one expert slot."""
        tasks = []
        for buffer_dict in (self.scale_cpu_buffers,
                            self.offset_cpu_buffers,
                            self.scale_bias_cpu_buffers):
            for attr_name, buffers in buffer_dict.items():
                local_eid = self._cpu_local(layer_idx, eid)
                dev_tensor = getattr(layer, attr_name, None)
                if dev_tensor is None or layer_idx >= len(buffers):
                    continue
                dst = dev_tensor.data[slot]
                if (not self.exclusive_shared_cpu_enabled
                        and local_eid is not None
                        and local_eid < len(buffers[layer_idx])):
                    src_tensor = buffers[layer_idx][local_eid]
                    source = src_tensor.reshape(dst.shape)
                    nbytes = src_tensor.numel() * src_tensor.element_size()
                elif (self.torch_shared_cpu_weights_enabled
                      or getattr(self._get_h2d_transport(),
                                 "supports_remote_sources", False)):
                    source = self._shared_h2d_source(
                        layer_idx, eid, attr_name)
                    nbytes = dst.numel() * dst.element_size()
                else:
                    continue
                tasks.append(H2DCopyTask(
                    source=source,
                    destination=dst,
                    nbytes=nbytes,
                    name=f"{attr_name}[L{layer_idx},E{eid}->S{slot}]",
                ))
        return tasks

    def _copy_quant_attrs_into_slot(self, layer, layer_idx, eid, slot):
        """Copy one expert's quant attributes through the H2D transport."""
        self._get_h2d_transport().copy_batch(
            self._build_quant_attr_h2d_tasks(layer, layer_idx, eid, slot))

    def _apply_multi_card_substitution(
        self,
        layer_idx,
        topk_ids_h,
        router_logits_h,
        log2phy_h,
        mc2_mask_h,
        scoring_func,
        correction_bias_h,
        cpu_group=None,
    ):
        """Atomically substitute source experts across all active EP rows."""
        if mc2_mask_h is None:
            active_rows = torch.arange(topk_ids_h.shape[0])
        else:
            active_rows = mc2_mask_h.bool().nonzero(
                as_tuple=True)[0]

        original_ids = topk_ids_h.index_select(
            0, active_rows)[:, :self.topk]
        active_logits = router_logits_h.index_select(0, active_rows)
        plan = plan_expert_substitutions(
            active_logits,
            original_ids,
            log2phy_h,
            expert_substitution_threshold=(
                self.offload_config.expert_substitution_threshold),
            scoring_func=scoring_func,
            e_score_correction_bias=correction_bias_h,
        )
        from vllm_ascend.expert_offload.multi_card_planner import (
            gather_global_substitution_state_cpu,
        )

        global_referenced, global_blocked = \
            gather_global_substitution_state_cpu(
                plan.referenced, plan.blocked, cpu_group)
        allowed = global_referenced & ~global_blocked
        substituted_ids = commit_expert_substitutions(
            plan,
            allowed,
            original_ids,
        )
        if self._debug:
            self._log_expert_substitution(
                layer_idx, original_ids, substituted_ids)
        updated_ids = topk_ids_h.index_select(0, active_rows)
        updated_ids[:, :self.topk].copy_(substituted_ids)
        topk_ids_h.index_copy_(0, active_rows, updated_ids)

    def _update_weights_multi_card(self, args):
        """Trace and run one multi-card decode callback or eager update."""
        if not self._debug:
            return self._update_weights_multi_card_impl(args)
        layer_idx = args[3]
        is_prefetch = bool(args[5])
        from_graph_callback = (
            (len(args) == 8 and bool(args[7]))
            or (len(args) == 13 and bool(args[12])))
        context = self._begin_mc_debug_callback(
            layer_idx, is_prefetch, from_graph_callback)
        context["stage"] = "callback_body"
        status = "ok"
        try:
            return self._update_weights_multi_card_impl(args, context)
        except BaseException as exc:
            status = f"error:{type(exc).__name__}"
            error_stage = (context or {}).get("stage", "unknown")
            self._log_mc_debug_event(
                "CB_ERROR",
                context,
                error_stage=error_stage,
                error_type=type(exc).__name__,
                error_message=repr(exc),
            )
            logger.exception(
                "[MC_DEBUG] callback traceback rank=%s layer=%s "
                "source=%s prefetch=%s stage=%s error=%r",
                self.ep_rank,
                layer_idx,
                context.get("source", "unknown"),
                is_prefetch,
                error_stage,
                exc,
            )
            raise
        finally:
            self._end_mc_debug_callback(context, status)

    def _update_weights_multi_card_impl(self, args, debug_context=None):
        """Host callback (graph replay) / inline (eager) for multi-card DECODE
        placement + H2D. Reads the pinned CPU topk_ids_h, does CPU bincount +
        gloo all_reduce (cpu_group) for global expert counts, plans the
        load-balanced placement, H2D-loads misses on load_stream (synced to gate
        the compute stream), and writes placement.log2phy into the pinned
        log2phy_h (the wrapper H2D-copies it back to the NPU tensor).
        """
        if len(args) in (7, 8):
            (topk_ids_h, log2phy_h, layer, layer_idx, per_rank_slots,
             is_prefetch, mc2_mask_h) = args[:7]
            do_substitution = False
            topk_weights_h = None
        else:
            (topk_ids_h, log2phy_h, layer, layer_idx, per_rank_slots,
             is_prefetch, mc2_mask_h, do_substitution, router_logits_h,
             scoring_func, correction_bias_h, topk_weights_h) = args[:12]
        decode_start = time.perf_counter() if self._debug else None
        from vllm.distributed.parallel_state import get_ep_group

        from vllm_ascend.expert_offload.multi_card_planner import plan_placement

        cpu_group = get_ep_group().cpu_group if self.ep_size > 1 else None

        # Substitute against the previous FULL global placement before global
        # counting. Each rank changes only its active local rows; the following
        # all-reduce makes the substituted route counts globally consistent.
        if do_substitution:
            if debug_context is not None:
                debug_context["stage"] = "substitution"
            self._apply_multi_card_substitution(
                layer_idx, topk_ids_h, router_logits_h, log2phy_h,
                mc2_mask_h, scoring_func, correction_bias_h, cpu_group)

        # Drop pad-token rows (mc2_mask==0) BEFORE any counting / placing / LRU.
        # Under single-batch TP the ranks past the real-token count hold PAD
        # tokens (zero hidden) whose topk is garbage; counting them inflates
        # global_counts (-> wrong placement + wasted H2D), corrupts the LRU
        # freq, and distorts hit/miss stats. An all-pad rank contributes an
        # empty [0, topk] view -> zero local counts; the all_reduce still
        # carries the real ranks' counts, and the global placement still
        # assigns that rank the real experts MC2 dispatches to it. mc2_mask_h
        # None -> all-active (backward compatible).
        if mc2_mask_h is not None:
            active_mask = mc2_mask_h.bool()
            topk_for_count = topk_ids_h[active_mask]
            weights_for_count = (topk_weights_h[active_mask]
                                 if topk_weights_h is not None else None)
        else:
            topk_for_count = topk_ids_h
            weights_for_count = topk_weights_h

        if self._debug:
            self._log_mc_router_observation(layer_idx, topk_for_count)

        if debug_context is not None:
            debug_context["stage"] = "count_and_hotness"
        counts_start = time.perf_counter() if self._debug else None
        global_counts, cache_on, hotness, prev_log2phy = \
            self._gather_global_counts_and_hotness(layer_idx, topk_for_count,
                                                    cpu_group,
                                                    weights_for_count,
                                                    debug_context)
        counts_ms = ((time.perf_counter() - counts_start) * 1000.0
                     if counts_start is not None else 0.0)

        # MemFabric SHARED exposes peer shard pointers, so published layers may
        # use global load-balanced placement. Unpublished late layers (MTP)
        # retain owner-shard placement until their pointers are available.
        if debug_context is not None:
            debug_context["stage"] = "shared_layer_ready"
        if (self.exclusive_shared_cpu_enabled
                and not self._shared_h2d_layer_ready(layer_idx)):
            raise RuntimeError(
                "Exclusive shared decode cannot run before the compact "
                f"shared CPU layer is finalized: layer={layer_idx}")
        force_shard = (
            getattr(self, '_shard_size', None)
            if (self.offload_config.shard_per_rank
                and not self._shared_h2d_layer_ready(layer_idx)) else None)
        if debug_context is not None:
            debug_context["stage"] = "placement"
        placement_start = time.perf_counter() if self._debug else None
        placement = plan_placement(global_counts, self.ep_size, per_rank_slots,
                                   prev_log2phy, hotness, force_shard=force_shard)
        placement_ms = ((time.perf_counter() - placement_start) * 1000.0
                        if placement_start is not None else 0.0)
        if cache_on and not self.exclusive_dynamic_enabled:
            self._mc_prev_log2phy[layer_idx] = placement.log2phy.clone()
        if self._debug:
            self._log_mc_decode_plan(
                layer_idx, global_counts, placement, per_rank_slots,
                counts_ms, placement_ms)

        # Communication selection should conservatively keep an overflowing
        # batch out of MC2.  Reaching this point means the admission invariant
        # was violated (or configuration changed after selection).  It is too
        # late to switch collectives safely: every rank has already entered the
        # MC2 execution path.  Fail explicitly instead of mapping experts to
        # unrelated weights via a spread log2phy and silently corrupting output.
        if debug_context is not None:
            debug_context["stage"] = "capacity_check"
        if placement.unassigned:
            layer_capacity = per_rank_slots * self.ep_size
            cpu_mode = (
                "torch_shared" if self.torch_shared_cpu_weights_enabled else
                "sharded" if self.offload_config.shard_per_rank else
                "replicated")
            sample = [int(eid) for eid in placement.unassigned[
                :self._DEBUG_EXPERT_SAMPLE_LIMIT]]
            raise RuntimeError(
                "multi-card expert-offload MC2 placement overflow: "
                f"rank={self.ep_rank} layer={layer_idx} cpu_mode={cpu_mode} "
                f"global_active={int((global_counts > 0).sum())} "
                f"global_capacity={layer_capacity} "
                f"per_rank_slots={per_rank_slots} "
                f"unassigned_count={len(placement.unassigned)} "
                f"unassigned_sample={sample}. The batch should have been "
                "routed to ALLTOALL by conservative MC2 admission."
            )

        my_experts = placement.per_rank_experts[self.ep_rank]
        # active_set = this step's token topk (the NEEDED experts). Only these
        # count in the hit/miss metric — retained-but-unneeded experts stay
        # cached but aren't counted as hits — matching single-card's
        # needed-based rate so multi vs single hit rates are comparable now
        # that placement retains a persistent hot set across steps.
        active_set = (set(global_counts.nonzero(as_tuple=True)[0].tolist())
                      if global_counts is not None else None)
        if debug_context is not None:
            debug_context["stage"] = "resident_diff"
        resident_map, hits, misses = self._compute_resident_hits(
            layer_idx, my_experts, cache_on, active_set)
        h2d_start = time.perf_counter() if self._debug and misses else None
        if self.exclusive_shared_cpu_enabled:
            if debug_context is not None:
                debug_context["stage"] = "shared_swap_lock"
            lock = self._exclusive_layer_locks[layer_idx]
            with lock:
                self._swap_expert_weights_multi_card_shared(
                    layer, layer_idx, placement, per_rank_slots,
                    log2phy_h, debug_context)
            resident_map = self._mc_resident[layer_idx]
        elif self.exclusive_sharded_cpu_enabled:
            if debug_context is not None:
                debug_context["stage"] = "sharded_swap"
            lock = self._exclusive_layer_locks[layer_idx]
            with lock:
                self._swap_expert_weights_multi_card_sharded(
                    layer, layer_idx, placement, per_rank_slots,
                    log2phy_h)
            resident_map = self._mc_resident[layer_idx]
        elif misses:
            if debug_context is not None:
                debug_context["stage"] = "h2d_load"
            self._h2d_load_mc_misses(layer, layer_idx, misses, resident_map,
                                     debug_context)
        h2d_ms = ((time.perf_counter() - h2d_start) * 1000.0
                  if h2d_start is not None else 0.0)
        # Write the FULL global placement into the pinned log2phy_h; the wrapper
        # H2D-copies it back to the NPU log2phy. Must be the FULL placement (not
        # just this rank) so MC2 routes tokens cross-rank correctly — writing only
        # my_experts would leave remote experts at -1 -> clamp 0 -> zero cross-
        # rank traffic -> MC2 uniform-mode dispatch deadlocks.
        if not self.exclusive_shared_cpu_enabled:
            if debug_context is not None:
                debug_context["stage"] = "publish_log2phy"
            log2phy_h.copy_(placement.log2phy)
        if self._debug:
            if debug_context is not None:
                debug_context["stage"] = "cache_log"
            total_ms = (time.perf_counter() - decode_start) * 1000.0
            self._log_mc_decode_cache(
                layer_idx, my_experts, hits, misses, resident_map,
                placement.log2phy, per_rank_slots, is_prefetch, h2d_ms,
                total_ms)

    def _log_mc_router_observation(self, layer_idx, topk_ids_h):
        if not self._debug or not logger.isEnabledFor(logging.DEBUG):
            return
        num_tokens = topk_ids_h.size(0)
        topk = topk_ids_h.size(1) if topk_ids_h.dim() > 1 else 1
        counts = Counter(int(e) for e in topk_ids_h.reshape(-1).tolist())
        hottest = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:8]
        logger.debug(
            "[MC_OBS] rank=%s L=%s router: tokens=%s topk=%s "
            "uniq_experts=%d/%d top8(expert:count)=%s",
            self.ep_rank, layer_idx, num_tokens, topk, len(counts),
            self.num_total_experts, hottest)

    def _log_mc_decode_plan(self, layer_idx, global_counts, placement,
                            per_rank_slots, counts_ms, placement_ms):
        """Log one compact copy of the deterministic global placement plan."""
        if not self._debug or self.ep_rank != 0:
            return
        active_ids = global_counts.nonzero(as_tuple=True)[0].tolist()
        active_per_rank = [0 for _ in range(self.ep_size)]
        for eid in active_ids:
            physical_id = int(placement.log2phy[int(eid)])
            if physical_id >= 0:
                rank = physical_id // per_rank_slots
                if 0 <= rank < self.ep_size:
                    active_per_rank[rank] += 1
        assigned_per_rank = [
            sum(int(eid) >= 0 for eid in experts)
            for experts in placement.per_rank_experts
        ]
        capacity_ok = (
            not placement.unassigned
            and all(count <= per_rank_slots for count in assigned_per_rank)
        )
        cpu_mode = (
            "torch_shared" if self.torch_shared_cpu_weights_enabled else
            "sharded" if self.offload_config.shard_per_rank else
            "replicated")
        unassigned = [int(eid) for eid in placement.unassigned]
        logger.info(
            "[MC_OBS] rank=%s L=%s DECODE plan: cpu_mode=%s "
            "global_routes=%d global_active=%d global_capacity=%d "
            "active_per_rank=%s assigned_per_rank=%s capacity_ok=%s "
            "global_counts_checksum=%d log2phy_checksum=%d "
            "unassigned_count=%d unassigned_sample=%s per_rank_load=%s "
            "timing_ms{counts_lrc=%.3f,placement=%.3f}",
            self.ep_rank, layer_idx, cpu_mode, int(global_counts.sum()),
            len(active_ids), per_rank_slots * self.ep_size, active_per_rank,
            assigned_per_rank, capacity_ok,
            _stable_int_checksum(global_counts),
            _stable_int_checksum(placement.log2phy),
            len(unassigned), unassigned[:self._DEBUG_EXPERT_SAMPLE_LIMIT],
            placement.per_rank_load,
            counts_ms, placement_ms)

    def _log_mc_lrc_state(self, layer_idx, global_counts,
                          global_router_score, hotness):
        """Log identical-on-every-rank LRC inputs/state for verification."""
        if not self._debug or self._mc_lrc is None:
            return
        state = self._mc_lrc.layer_states[layer_idx]
        active_ids = global_counts.nonzero(as_tuple=True)[0].tolist()
        sample_limit = self._DEBUG_EXPERT_SAMPLE_LIMIT
        hottest = sorted(
            active_ids,
            key=lambda eid: (-float(hotness[eid]), int(eid)),
        )[:sample_limit]
        hot_sample = [
            {
                "expert": int(eid),
                "count": int(global_counts[eid]),
                "freq": int(state.freq[eid]),
                "ema": round(float(state.ema[eid]), 6),
                "router": round(float(state.router_score[eid]), 6),
                "hotness": round(float(hotness[eid]), 6),
            }
            for eid in hottest
        ]
        router_input_checksum = (
            _stable_float_checksum(global_router_score)
            if global_router_score is not None else None)
        logger.info(
            "[MC_LRC] rank=%s/%s L=%s step=%s global_routes=%s "
            "global_active=%s counts_checksum=%s freq_sum=%s "
            "freq_checksum=%s ema_checksum=%s router_enabled=%s "
            "router_input_checksum=%s router_state_checksum=%s "
            "hotness_checksum=%s top_hot=%s",
            self.ep_rank, self.ep_size, layer_idx, state.step,
            int(global_counts.sum()), len(active_ids),
            _stable_int_checksum(global_counts), sum(state.freq),
            _stable_int_checksum(state.freq),
            _stable_float_checksum(state.ema),
            global_router_score is not None, router_input_checksum,
            _stable_float_checksum(state.router_score),
            _stable_float_checksum(hotness), hot_sample)

    def _sanitize_mc_router_scores(self, layer_idx, scores, expert_ids=None,
                                   stage="local_topk", debug_context=None):
        """Keep non-finite router scores out of the optional LRC heuristic.

        ``scores`` and ``expert_ids`` are CPU tensors copied for placement
        accounting; the router output used by model execution is untouched.
        Logging local values before Gloo identifies the originating rank,
        layer, route position and expert. A second call after Gloo protects
        against non-finite reduction results as well.
        """
        finite = torch.isfinite(scores)
        if bool(finite.all()):
            return scores

        invalid_positions = (~finite).nonzero(as_tuple=True)[0]
        nan_count = int(torch.isnan(scores).sum())
        infinite = torch.isinf(scores)
        posinf_count = int((infinite & (scores > 0)).sum())
        neginf_count = int((infinite & (scores < 0)).sum())
        sample = []
        for position in invalid_positions[
                :self._DEBUG_EXPERT_SAMPLE_LIMIT].tolist():
            number = float(scores[position])
            value = ("nan" if math.isnan(number) else
                     "+inf" if number > 0 else "-inf")
            expert_id = (int(expert_ids[position])
                         if expert_ids is not None else int(position))
            sample.append({
                "position": int(position),
                "expert": expert_id,
                "value": value,
            })
        source = (debug_context or {}).get("source", "-")
        logger.warning(
            "[MC_ROUTER_NONFINITE] rank=%s/%s layer=%s stage=%s "
            "source=%s total=%s nonfinite=%s nan=%s posinf=%s neginf=%s "
            "action=zero_for_lrc sample=%s",
            self.ep_rank, self.ep_size, layer_idx, stage, source,
            scores.numel(), invalid_positions.numel(), nan_count,
            posinf_count, neginf_count, sample)
        return scores.masked_fill(~finite, 0.0)

    def _gather_global_counts_and_hotness(self, layer_idx, topk_ids_h,
                                          cpu_group, topk_weights_h=None,
                                          debug_context=None):
        """CPU bincount + gloo all_reduce -> global expert counts, then update
        the LRC hotness policy (the same one single-card uses: recent freq +
        EMA + age + optional router score) from the real GLOBAL route counts
        and return its per-expert hotness. global_counts and router score sums
        are all-reduced EVERY step (identical across ranks after the mc2_mask
        filter), so the LRC state + hotness are too -> placement/eviction stay
        deterministic. Gloo runs on cpu_group rather than the captured NPU
        stream, but all callbacks must still enter its collectives in the same
        order on every rank."""
        from vllm_ascend.expert_offload.multi_card_planner import (
            local_expert_counts_cpu)
        local_counts = local_expert_counts_cpu(topk_ids_h, self.num_total_experts)
        global_counts = self._gather_cpu_with_mc_debug(
            local_counts, cpu_group, "count", debug_context)
        cache_on = self.offload_config.cache_policy_enabled
        if not cache_on:
            retain_exclusive = self.exclusive_dynamic_enabled
            return (global_counts, retain_exclusive, None,
                    self._mc_prev_log2phy.get(layer_idx)
                    if retain_exclusive else None)
        # Reuse the configured single-card policy so every cache tuning knob
        # has identical meaning in multi-card mode.  The lazy fallback only
        # protects unusual unit-test/partial-initialization paths.
        if self._mc_lrc is None:
            self._mc_lrc = self.cache_policy
        if self._mc_lrc is None:
            return (global_counts, False, None,
                    self._mc_prev_log2phy.get(layer_idx))
        while len(self._mc_lrc.layer_states) <= layer_idx:
            self._mc_lrc.add_layer()

        global_router_score = None
        if topk_weights_h is not None:
            ids = topk_ids_h.reshape(-1).to(torch.int64)
            scores = topk_weights_h.reshape(-1).to(torch.float32)
            scores = self._sanitize_mc_router_scores(
                layer_idx, scores, ids, "local_topk", debug_context)
            local_score_sum = torch.zeros(self.num_total_experts,
                                          dtype=torch.float32)
            local_score_sum.scatter_add_(0, ids, scores)
            global_score_sum = self._gather_cpu_with_mc_debug(
                local_score_sum, cpu_group, "router_score", debug_context)
            global_score_sum = self._sanitize_mc_router_scores(
                layer_idx, global_score_sum, stage="global_sum",
                debug_context=debug_context)
            global_router_score = torch.zeros_like(global_score_sum)
            active = global_counts > 0
            global_router_score[active] = (
                global_score_sum[active] /
                global_counts[active].to(global_score_sum.dtype))
        self._mc_lrc.observe_global_counts(
            layer_idx, global_counts, global_router_score)
        hotness = self._mc_lrc.hotness_array(layer_idx)
        self._log_mc_lrc_state(
            layer_idx, global_counts, global_router_score, hotness)
        return (global_counts, cache_on, hotness,
                self._mc_prev_log2phy.get(layer_idx))

    def _compute_resident_hits(self, layer_idx, my_experts, cache_on,
                               active_set=None):
        """Split this rank's ACTIVE placed experts into cache hits (expert
        already resident in its assigned slot) vs misses (need H2D). Returns
        (resident_map, hits, misses).

        Only ACTIVE experts (in ``active_set`` = this step's token topk) are
        counted: retained-but-not-needed experts stay cached but don't count
        as hits, mirroring single-card's (needed ∩ on_device)/needed so the
        hit rate is comparable across configs. ``active_set=None`` counts all
        (backward-compatible fallback)."""
        if not cache_on:
            # No cache: every (active) expert is a miss (full H2D every step).
            misses = [(s, int(e)) for s, e in enumerate(my_experts)
                      if e >= 0 and (active_set is None or int(e) in active_set)]
            return {}, [], misses
        resident_map = self._mc_resident.setdefault(layer_idx, {})
        hits, misses = [], []
        for slot, eid in enumerate(my_experts):
            if eid < 0:
                continue
            eid = int(eid)
            if active_set is not None and eid not in active_set:
                continue  # retained but not needed this step: cached, not counted
            (hits if resident_map.get(slot) == eid else misses).append((slot, eid))
        return resident_map, hits, misses

    def _get_h2d_transport(self):
        """Return the transport, lazily initializing collective SHARED."""
        if not hasattr(self, 'h2d_transport'):
            self.h2d_transport = TorchCopyH2DTransport()
        elif self.h2d_transport is None:
            self.h2d_transport = self._create_h2d_transport()
        return self.h2d_transport

    def _allocate_expert_host_tensor(self, shape, dtype) -> torch.Tensor:
        """Allocate weight/quant storage owned by the selected H2D backend."""
        return self._get_h2d_transport().allocate_host_tensor(shape, dtype)

    def _synchronize_h2d(self) -> None:
        """Wait for the load stream and retire backend copy descriptors."""
        self._get_h2d_transport().synchronize(self.load_stream)

    def close(self) -> None:
        """Release H2D backend resources; safe to call more than once."""
        transport = getattr(self, 'h2d_transport', None)
        if transport is not None:
            transport.synchronize(self.load_stream)
            transport.close()
        shared_pool = getattr(self, '_torch_shared_cpu_pool', None)
        if shared_pool is not None:
            shared_pool.close()
        prefill_pyhccl = getattr(
            self, '_exclusive_prefill_pyhccl', None)
        if prefill_pyhccl is not None:
            prefill_pyhccl.close()
            self._exclusive_prefill_pyhccl = None
        comm_buffers = getattr(
            self, '_exclusive_prefill_comm_buffers', None)
        if comm_buffers is not None:
            comm_buffers.clear()

    def _build_expert_h2d_tasks(self, layer, layer_idx, eid,
                                slot) -> list[H2DCopyTask]:
        """Build format-preserving weight and quant H2D tasks for one expert."""
        w13_start = slot * self.w13_expert_size_bytes
        w2_start = slot * self.w2_expert_size_bytes
        w13_dst = _expert_weight(layer, "w13_weight").data.untyped_storage()[
            w13_start:w13_start + self.w13_expert_size_bytes]
        w2_dst = _expert_weight(layer, "w2_weight").data.untyped_storage()[
            w2_start:w2_start + self.w2_expert_size_bytes]
        tasks = [
            H2DCopyTask(
                source=self._expert_src_storage(layer_idx, eid, 'w13'),
                destination=w13_dst,
                nbytes=self.w13_expert_size_bytes,
                name=f"w13[L{layer_idx},E{eid}->S{slot}]",
            ),
            H2DCopyTask(
                source=self._expert_src_storage(layer_idx, eid, 'w2'),
                destination=w2_dst,
                nbytes=self.w2_expert_size_bytes,
                name=f"w2[L{layer_idx},E{eid}->S{slot}]",
            ),
        ]
        tasks.extend(
            self._build_quant_attr_h2d_tasks(layer, layer_idx, eid, slot))
        return tasks

    def _iter_cpu_expert_buffers(self, layer_idx: int):
        """Yield every canonical CPU buffer participating in a swap."""
        if self.exclusive_shared_cpu_enabled:
            yield "w13", self._torch_shared_cpu_buffers[(layer_idx, "w13")]
            yield "w2", self._torch_shared_cpu_buffers[(layer_idx, "w2")]
            for buffer_dict in (self.scale_cpu_buffers,
                                self.offset_cpu_buffers,
                                self.scale_bias_cpu_buffers):
                for attr_name, layer_buffers in buffer_dict.items():
                    if (layer_idx < len(layer_buffers)
                            and (layer_idx, attr_name)
                            in self._torch_shared_cpu_buffers):
                        yield attr_name, self._torch_shared_cpu_buffers[
                            (layer_idx, attr_name)]
            return
        yield "w13", self.w13_weights_cpu[layer_idx]
        yield "w2", self.w2_weights_cpu[layer_idx]
        for buffer_dict in (self.scale_cpu_buffers,
                            self.offset_cpu_buffers,
                            self.scale_bias_cpu_buffers):
            for attr_name, layer_buffers in buffer_dict.items():
                if (layer_idx < len(layer_buffers)
                        and layer_buffers[layer_idx]):
                    yield attr_name, layer_buffers[layer_idx]

    def _allocate_exclusive_swap_bundle(
            self, layer_idx: int, cpu_slot: int) -> dict[str, torch.Tensor]:
        """Allocate transient pinned storage for one evicted expert.

        A miss cannot overwrite its incoming CPU buffer until that buffer has
        finished H2D.  One transient bundle per simultaneously swapped expert
        is therefore required.  Bundles are transaction-local, so canonical
        storage remains exactly ``CPU experts + NPU experts == all experts``
        and the extra peak is limited to misses of the active layer.
        """
        bundle = {}
        for name, buffers in self._iter_cpu_expert_buffers(layer_idx):
            template = buffers[cpu_slot]
            bundle[name] = self._allocate_expert_host_tensor(
                template.shape, template.dtype)
        return bundle

    def _build_expert_d2h_tasks(
        self,
        layer,
        layer_idx: int,
        victim_eid: int,
        victim_slot: int,
        scratch: dict[str, torch.Tensor],
    ) -> list[H2DCopyTask]:
        """Build device-to-host copies for one evicted expert."""
        tasks = [
            H2DCopyTask(
                source=self._expert_device_storage(
                    layer, victim_slot, "w13"),
                destination=scratch["w13"].untyped_storage(),
                nbytes=self.w13_expert_size_bytes,
                name=(f"swap-d2h-w13[L{layer_idx},E{victim_eid}"
                      f"<-S{victim_slot}]"),
                direction=CopyDirection.D2H,
            ),
            H2DCopyTask(
                source=self._expert_device_storage(
                    layer, victim_slot, "w2"),
                destination=scratch["w2"].untyped_storage(),
                nbytes=self.w2_expert_size_bytes,
                name=(f"swap-d2h-w2[L{layer_idx},E{victim_eid}"
                      f"<-S{victim_slot}]"),
                direction=CopyDirection.D2H,
            ),
        ]
        for buffer_dict in (self.scale_cpu_buffers,
                            self.offset_cpu_buffers,
                            self.scale_bias_cpu_buffers):
            for attr_name, layer_buffers in buffer_dict.items():
                if (layer_idx >= len(layer_buffers)
                        or attr_name not in scratch):
                    continue
                dev_tensor = getattr(layer, attr_name, None)
                if dev_tensor is None:
                    continue
                source = dev_tensor.data[victim_slot]
                destination = scratch[attr_name].reshape(source.shape)
                tasks.append(H2DCopyTask(
                    source=source,
                    destination=destination,
                    nbytes=source.numel() * source.element_size(),
                    name=(f"swap-d2h-{attr_name}[L{layer_idx},"
                          f"E{victim_eid}<-S{victim_slot}]"),
                    direction=CopyDirection.D2H,
                ))
        return tasks

    def _commit_exclusive_swap(
        self,
        layer_idx: int,
        log2phy_np,
        swap,
        scratch: dict[str, torch.Tensor],
        physical_slot_base: int = 0,
    ) -> None:
        """Commit one synchronized swap to CPU buffers and ownership maps."""
        slot, incoming_eid, victim_eid, cpu_slot = swap
        for name, buffers in self._iter_cpu_expert_buffers(layer_idx):
            buffers[cpu_slot] = scratch[name]

        self._cpu_slot_to_eid[layer_idx][cpu_slot] = victim_eid
        self._eid_to_cpu_slot[layer_idx][incoming_eid] = -1
        self._eid_to_cpu_slot[layer_idx][victim_eid] = cpu_slot
        self._npu_slot_to_eid[layer_idx][slot] = incoming_eid
        self._eid_to_npu_slot[layer_idx][victim_eid] = -1
        self._eid_to_npu_slot[layer_idx][incoming_eid] = slot
        log2phy_np[victim_eid] = -1
        log2phy_np[incoming_eid] = physical_slot_base + slot

    def _swap_expert_weights(self, layer, layer_idx: int, swaps,
                             log2phy_np, physical_slot_base: int = 0) -> None:
        """Execute a batch of exclusive CPU↔NPU swaps atomically.

        All victim D2H copies are queued before any incoming H2D copy.  The
        load stream then refreshes derived fp32 scales and synchronizes once.
        Only after that barrier do the canonical CPU buffer references,
        inverse ownership maps and log2phy change.
        """
        swaps = list(swaps)
        if not swaps:
            return
        if len({swap[0] for swap in swaps}) != len(swaps):
            raise RuntimeError(
                f"Duplicate NPU slots in exclusive swap plan: {swaps}")
        if len({swap[3] for swap in swaps}) != len(swaps):
            raise RuntimeError(
                f"Duplicate CPU slots in exclusive swap plan: {swaps}")

        scratch_bundles = [
            self._allocate_exclusive_swap_bundle(layer_idx, cpu_slot)
            for _, _, _, cpu_slot in swaps
        ]
        d2h_tasks = []
        h2d_tasks = []
        for (slot, incoming_eid, victim_eid, _), scratch in zip(
                swaps, scratch_bundles):
            d2h_tasks.extend(self._build_expert_d2h_tasks(
                layer, layer_idx, victim_eid, slot, scratch))
            h2d_tasks.extend(self._build_expert_h2d_tasks(
                layer, layer_idx, incoming_eid, slot))

        self._get_h2d_transport().copy_batch(d2h_tasks + h2d_tasks)
        for slot, _, _, _ in swaps:
            self._refresh_expert_fp32_scale(layer, slot)
        self._synchronize_h2d()

        for swap, scratch in zip(swaps, scratch_bundles):
            self._commit_exclusive_swap(
                layer_idx, log2phy_np, swap, scratch,
                physical_slot_base=physical_slot_base)
        self._validate_exclusive_ownership(layer_idx)

    def _swap_expert_weights_multi_card_sharded(
        self,
        layer,
        layer_idx: int,
        placement,
        per_rank_slots: int,
        log2phy_h,
    ) -> None:
        """Apply an owner-constrained placement using local swaps only."""
        my_experts = list(placement.per_rank_experts[self.ep_rank])
        if (len(my_experts) != per_rank_slots
                or any(int(eid) < 0 for eid in my_experts)):
            raise RuntimeError(
                "Rank-local exclusive placement must keep every local NPU "
                "slot populated: "
                f"rank={self.ep_rank}, layer={layer_idx}, "
                f"expected_slots={per_rank_slots}, experts={my_experts}")
        local_start = self._shard_base
        local_end = local_start + self._shard_size
        swaps = []
        for slot, incoming_eid in enumerate(my_experts):
            incoming_eid = int(incoming_eid)
            if not local_start <= incoming_eid < local_end:
                raise RuntimeError(
                    "Rank-local exclusive placement selected a remote expert: "
                    f"rank={self.ep_rank}, layer={layer_idx}, "
                    f"expert={incoming_eid}, shard=[{local_start},{local_end})")
            victim_eid = self._npu_slot_to_eid[layer_idx][slot]
            if incoming_eid == victim_eid:
                continue
            cpu_slot = self._eid_to_cpu_slot[layer_idx][incoming_eid]
            if cpu_slot < 0:
                raise RuntimeError(
                    "Rank-local exclusive placement attempted a non-CPU "
                    "incoming expert: "
                    f"rank={self.ep_rank}, layer={layer_idx}, "
                    f"expert={incoming_eid}")
            swaps.append((slot, incoming_eid, int(victim_eid), cpu_slot))

        physical_slot_base = self.ep_rank * per_rank_slots
        with torch_npu.npu.stream(self.load_stream):
            self._swap_expert_weights(
                layer, layer_idx, swaps, log2phy_h.numpy(),
                physical_slot_base=physical_slot_base)
        log2phy_h.copy_(placement.log2phy)
        self._mc_prev_log2phy[layer_idx] = placement.log2phy.clone()
        self._mc_resident[layer_idx] = {
            slot: int(eid)
            for slot, eid in enumerate(self._npu_slot_to_eid[layer_idx])
        }
        if self._debug and swaps:
            logger.info(
                "[EXCLUSIVE-SHARDED-SWAP] layer=%d rank=%d swaps=%d "
                "cross_rank_weight_transfer=False",
                layer_idx, self.ep_rank, len(swaps))

    def _exclusive_shared_swaps_from_placement(
        self,
        layer_idx: int,
        placement,
        per_rank_slots: int,
    ) -> list[tuple[int, int, int, int]]:
        """Diff a deterministic global placement into CPU/NPU swaps."""
        target_by_slot = [-1] * (per_rank_slots * self.ep_size)
        for rank, experts in enumerate(placement.per_rank_experts):
            for local_slot, eid in enumerate(experts):
                if int(eid) >= 0:
                    target_by_slot[rank * per_rank_slots + local_slot] = int(eid)
        swaps = []
        for physical_slot, incoming_eid in enumerate(target_by_slot):
            if incoming_eid < 0:
                continue
            victim_eid = self._npu_slot_to_eid[layer_idx][physical_slot]
            if incoming_eid == victim_eid:
                continue
            cpu_slot = self._eid_to_cpu_slot[layer_idx][incoming_eid]
            if cpu_slot < 0:
                raise RuntimeError(
                    "Exclusive placement attempted to move an NPU resident "
                    "instead of retaining its physical slot: "
                    f"layer={layer_idx}, expert={incoming_eid}, "
                    f"old_physical={self._eid_to_npu_slot[layer_idx][incoming_eid]}, "
                    f"new_physical={physical_slot}")
            swaps.append((physical_slot, incoming_eid,
                          int(victim_eid), cpu_slot))
        if len({swap[3] for swap in swaps}) != len(swaps):
            raise RuntimeError(
                f"Duplicate compact CPU slots in global swap plan: {swaps}")
        return swaps

    @staticmethod
    def _raise_distributed_swap_errors(stage: str, errors) -> None:
        failures = [error for error in errors if error is not None]
        if failures:
            raise RuntimeError(
                f"Exclusive shared swap failed during {stage}: "
                + "; ".join(failures))

    def _swap_expert_weights_multi_card_shared(
        self,
        layer,
        layer_idx: int,
        placement,
        per_rank_slots: int,
        log2phy_h,
        debug_context=None,
    ) -> None:
        """Atomically swap this rank's slots and commit global ownership.

        Each rank performs only the D2H/H2D operations for its physical NPU
        slots. Device-copy and shared-CPU-commit status are gathered before
        any ownership map changes, so all ranks either publish the same new
        mapping or fail closed.
        """
        from torch import distributed as dist
        from vllm.distributed.parallel_state import get_ep_group

        cpu_group = get_ep_group().cpu_group
        if debug_context is not None:
            debug_context["stage"] = "shared_swap_plan_diff"
        global_swaps = self._exclusive_shared_swaps_from_placement(
            layer_idx, placement, per_rank_slots)
        base = self.ep_rank * per_rank_slots
        local_swaps = [
            (physical_slot - base, incoming_eid, victim_eid, cpu_slot)
            for physical_slot, incoming_eid, victim_eid, cpu_slot
            in global_swaps
            if base <= physical_slot < base + per_rank_slots
        ]
        if self._debug:
            sample_limit = getattr(
                self, "_DEBUG_EXPERT_SAMPLE_LIMIT", 8)
            self._log_mc_debug_event(
                "SHARED_SWAP_PLAN",
                debug_context,
                global_swaps=len(global_swaps),
                local_swaps=len(local_swaps),
                swap_sample=global_swaps[:sample_limit],
            )
        scratch_bundles = []
        local_error = None
        if debug_context is not None:
            debug_context["stage"] = "shared_swap_device_copy"
        try:
            scratch_bundles = [
                self._allocate_exclusive_swap_bundle(layer_idx, cpu_slot)
                for _, _, _, cpu_slot in local_swaps
            ]
            if local_swaps:
                d2h_tasks = []
                h2d_tasks = []
                for swap, scratch in zip(local_swaps, scratch_bundles):
                    local_slot, incoming_eid, victim_eid, _ = swap
                    d2h_tasks.extend(self._build_expert_d2h_tasks(
                        layer, layer_idx, victim_eid, local_slot, scratch))
                    h2d_tasks.extend(self._build_expert_h2d_tasks(
                        layer, layer_idx, incoming_eid, local_slot))
                with torch_npu.npu.stream(self.load_stream):
                    self._get_h2d_transport().copy_batch(
                        d2h_tasks + h2d_tasks)
                    for local_slot, _, _, _ in local_swaps:
                        self._refresh_expert_fp32_scale(layer, local_slot)
                    self._synchronize_h2d()
        except BaseException as exc:
            local_error = (
                f"rank={self.ep_rank} {type(exc).__name__}: {exc}")
            self._log_mc_debug_event(
                "SHARED_SWAP_LOCAL_ERROR",
                debug_context,
                error_stage="device_copy",
                error_type=type(exc).__name__,
                error_message=repr(exc),
            )
            if self._debug:
                logger.exception(
                    "[EXCLUSIVE-SHARED-SWAP] local traceback rank=%s "
                    "layer=%s stage=device_copy error=%r",
                    self.ep_rank,
                    layer_idx,
                    exc,
                )
        device_errors = [None] * self.ep_size
        if debug_context is not None:
            debug_context["stage"] = "shared_swap_device_consensus"
        dist.all_gather_object(
            device_errors, local_error, group=cpu_group)
        self._raise_distributed_swap_errors("device copy", device_errors)

        local_error = None
        if debug_context is not None:
            debug_context["stage"] = "shared_swap_cpu_commit"
        try:
            for (_, _, _, cpu_slot), scratch in zip(
                    local_swaps, scratch_bundles):
                for name, buffers in self._iter_cpu_expert_buffers(layer_idx):
                    buffers[cpu_slot].copy_(scratch[name])
        except BaseException as exc:
            local_error = (
                f"rank={self.ep_rank} {type(exc).__name__}: {exc}")
            self._log_mc_debug_event(
                "SHARED_SWAP_LOCAL_ERROR",
                debug_context,
                error_stage="shared_cpu_commit",
                error_type=type(exc).__name__,
                error_message=repr(exc),
            )
            if self._debug:
                logger.exception(
                    "[EXCLUSIVE-SHARED-SWAP] local traceback rank=%s "
                    "layer=%s stage=shared_cpu_commit error=%r",
                    self.ep_rank,
                    layer_idx,
                    exc,
                )
        cpu_errors = [None] * self.ep_size
        if debug_context is not None:
            debug_context["stage"] = "shared_swap_cpu_consensus"
        dist.all_gather_object(cpu_errors, local_error, group=cpu_group)
        self._raise_distributed_swap_errors("shared CPU commit", cpu_errors)

        if debug_context is not None:
            debug_context["stage"] = "shared_swap_ownership_commit"
        for physical_slot, incoming_eid, victim_eid, cpu_slot in global_swaps:
            self._cpu_slot_to_eid[layer_idx][cpu_slot] = victim_eid
            self._eid_to_cpu_slot[layer_idx][incoming_eid] = -1
            self._eid_to_cpu_slot[layer_idx][victim_eid] = cpu_slot
            self._npu_slot_to_eid[layer_idx][physical_slot] = incoming_eid
            self._eid_to_npu_slot[layer_idx][victim_eid] = -1
            self._eid_to_npu_slot[layer_idx][incoming_eid] = physical_slot
        self._mc_prev_log2phy[layer_idx] = placement.log2phy.clone()
        my_experts = placement.per_rank_experts[self.ep_rank]
        self._mc_resident[layer_idx] = {
            slot: int(eid) for slot, eid in enumerate(my_experts)
            if int(eid) >= 0
        }
        log2phy_h.copy_(placement.log2phy)
        if debug_context is not None:
            debug_context["stage"] = "shared_swap_ownership_validate"
        self._validate_exclusive_ownership(layer_idx)
        if debug_context is not None:
            debug_context["stage"] = "shared_swap_complete"
        if self._debug and global_swaps:
            logger.info(
                "[EXCLUSIVE-SHARED-SWAP] layer=%d rank=%d "
                "global_swaps=%d local_swaps=%d",
                layer_idx, self.ep_rank, len(global_swaps), len(local_swaps))

    def _write_exclusive_log2phy(self, layer_idx: int, log2phy_np) -> None:
        """Rebuild staged log2phy from the authoritative ownership map."""
        log2phy_np.fill(-1)
        for slot, eid in enumerate(self._npu_slot_to_eid[layer_idx]):
            log2phy_np[eid] = slot

    @staticmethod
    def _refresh_expert_fp32_scale(layer, slot):
        if hasattr(layer, 'w13_weight_scale_fp32'):
            layer.w13_weight_scale_fp32[slot].copy_(
                layer.w13_weight_scale.data[slot].to(torch.float32))

    def _load_expert_weights_into_slots(self, layer, layer_idx, loads):
        """Batch H2D-copy ``(slot, eid)`` loads, then refresh derived data."""
        loads = list(loads)
        tasks = []
        for slot, eid in loads:
            tasks.extend(
                self._build_expert_h2d_tasks(layer, layer_idx, eid, slot))
        self._get_h2d_transport().copy_batch(tasks)
        for slot, _ in loads:
            self._refresh_expert_fp32_scale(layer, slot)

    def _load_expert_weights_into_slot(self, layer, layer_idx, eid, slot):
        """Compatibility wrapper for loading one expert into one device slot."""
        self._load_expert_weights_into_slots(layer, layer_idx, [(slot, eid)])

    def _h2d_load_mc_misses(self, layer, layer_idx, misses, resident_map,
                            debug_context=None):
        """H2D-load missed experts (w13/w2 + quant attrs) into their slots on
        load_stream, then synchronize to gate the compute stream."""
        if (self._debug and self._shared_h2d_layer_ready(layer_idx)
                and misses):
            remote_count = sum(
                self._shard_local(eid) is None for _, eid in misses)
            shared_backend = (
                "TORCH-SHARED-CPU-H2D"
                if self.torch_shared_cpu_weights_enabled
                else "MEMFABRIC-SHARED-H2D")
            logger.info(
                "[%s] layer=%d rank=%d loads=%d "
                "remote=%d local=%d",
                shared_backend, layer_idx, self.ep_rank, len(misses),
                remote_count,
                len(misses) - remote_count)
        if not self._debug:
            with torch_npu.npu.stream(self.load_stream):
                self._load_expert_weights_into_slots(
                    layer, layer_idx, misses)
                for slot, eid in misses:
                    resident_map[slot] = eid
                self._synchronize_h2d()
            return
        start_ns = time.perf_counter_ns()
        self._log_mc_debug_event(
            "H2D_ENTER", debug_context, misses=len(misses))
        try:
            with torch_npu.npu.stream(self.load_stream):
                self._load_expert_weights_into_slots(
                    layer, layer_idx, misses)
                for slot, eid in misses:
                    resident_map[slot] = eid
                self._log_mc_debug_event(
                    "H2D_SYNC_ENTER", debug_context, misses=len(misses))
                self._synchronize_h2d()
                self._log_mc_debug_event(
                    "H2D_SYNC_EXIT", debug_context, misses=len(misses))
        except BaseException as exc:
            self._log_mc_debug_event(
                "H2D_ERROR",
                debug_context,
                misses=len(misses),
                error_type=type(exc).__name__,
            )
            raise
        elapsed_us = (time.perf_counter_ns() - start_ns) // 1000
        self._log_mc_debug_event(
            "H2D_EXIT",
            debug_context,
            misses=len(misses),
            elapsed_us=elapsed_us,
        )

    def _log_mc_decode_cache(self, layer_idx, my_experts, hits, misses,
                             resident_map, log2phy, per_rank_slots,
                             is_prefetch=False, h2d_ms=0.0, total_ms=0.0):
        if not self._debug:
            return
        expected = {
            slot: int(eid)
            for slot, eid in enumerate(my_experts) if int(eid) >= 0
        }
        resident_mismatches = {
            slot: {"expected": eid, "resident": resident_map.get(slot)}
            for slot, eid in expected.items()
            if resident_map.get(slot) != eid
        }
        mapping_mismatches = {
            eid: {
                "expected_physical": self.ep_rank * per_rank_slots + slot,
                "actual_physical": int(log2phy[eid]),
            }
            for slot, eid in expected.items()
            if int(log2phy[eid]) != self.ep_rank * per_rank_slots + slot
        }
        requests = len(hits) + len(misses)
        hit_rate = len(hits) / requests if requests else 1.0
        cpu_mode = (
            "torch_shared" if self.torch_shared_cpu_weights_enabled else
            "sharded" if self.offload_config.shard_per_rank else
            "replicated")
        if not misses and not resident_mismatches and not mapping_mismatches:
            return
        sample_limit = self._DEBUG_EXPERT_SAMPLE_LIMIT
        logger.info(
            "[MC_OBS] rank=%s L=%s DECODE cache: cpu_mode=%s placed=%d "
            "hit=%d miss=%d hit_rate=%.4f resident_ok=%s mapping_ok=%s "
            "h2d_load_sample=%s prefetch=%s "
            "timing_ms{h2d=%.3f,total=%.3f}",
            self.ep_rank, layer_idx, cpu_mode, len(expected), len(hits),
            len(misses), hit_rate, not resident_mismatches,
            not mapping_mismatches, misses[:sample_limit], is_prefetch,
            h2d_ms, total_ms)
        if resident_mismatches or mapping_mismatches:
            logger.warning(
                "[MC_OBS] rank=%s L=%s DECODE cache mismatch: "
                "resident_sample=%s mapping_sample=%s",
                self.ep_rank, layer_idx,
                list(resident_mismatches.items())[:sample_limit],
                list(mapping_mismatches.items())[:sample_limit])

    def _log_expert_substitution(
        self,
        layer_idx: int,
        original_ids: torch.Tensor,
        substituted_ids: torch.Tensor,
    ) -> None:
        """Log CPU-side expert replacements when offload debug is enabled."""
        if not self._debug:
            return

        changed = original_ids != substituted_ids
        changed_positions = changed.nonzero(as_tuple=False)
        replacements = [
            {
                "token": int(token_idx),
                "position": int(position),
                "original": int(original_ids[token_idx, position]),
                "substitute": int(substituted_ids[token_idx, position]),
            }
            for token_idx, position in changed_positions.tolist()
        ]
        if not replacements:
            return
        sample_limit = self._DEBUG_EXPERT_SAMPLE_LIMIT
        logger.info(
            "[SUBST] layer=%d replacement_count=%d threshold=%.4f "
            "replacement_sample=%s truncated=%d",
            layer_idx,
            len(replacements),
            self.offload_config.expert_substitution_threshold,
            replacements[:sample_limit],
            max(0, len(replacements) - sample_limit),
        )

    @staticmethod
    def _ordered_unique_experts(topk_ids_h) -> list[int]:
        """Flatten routed experts while retaining first-seen order."""
        ordered = []
        seen = set()
        for eid in topk_ids_h.reshape(-1).tolist():
            eid = int(eid)
            if eid >= 0 and eid not in seen:
                seen.add(eid)
                ordered.append(eid)
        return ordered

    def _update_weights_exclusive_dynamic(
        self,
        topk_ids_h,
        log2phy_np,
        layer,
        layer_idx: int,
        topk_weights_h,
        is_prefetch: bool,
    ) -> None:
        """Single-card decode/prefetch planner for exclusive ownership."""
        lock = self._exclusive_layer_locks[layer_idx]
        with lock:
            with torch_npu.npu.stream(self.load_stream):
                # CPU/NPU ownership is authoritative.  Rebuild the staged map
                # before planning so a delayed host callback cannot reintroduce
                # an older mapping snapshot.
                self._write_exclusive_log2phy(layer_idx, log2phy_np)
                ordered = self._ordered_unique_experts(topk_ids_h)
                if not is_prefetch and self.cache_policy is not None:
                    router_scores = (
                        topk_weights_h.tolist()
                        if topk_weights_h is not None else None)
                    needed = self.cache_policy.observe(
                        layer_idx,
                        topk_ids_h.tolist(),
                        router_scores=router_scores,
                    )
                else:
                    needed = set(ordered)

                slot_owner = {
                    slot: eid for slot, eid in enumerate(
                        self._npu_slot_to_eid[layer_idx])
                }
                on_device = set(slot_owner.values())
                ordered_misses = [
                    eid for eid in ordered if eid not in on_device]
                if is_prefetch:
                    ordered_misses = ordered_misses[:self.prefetch_topk]
                else:
                    # LRC observe currently returns the routed set.  Include
                    # any future policy-added experts deterministically.
                    ordered_misses.extend(sorted(
                        (needed - on_device) - set(ordered_misses)))
                need_to_load = set(ordered_misses)
                already_there = needed & on_device

                if self.cache_policy is not None:
                    self._record_cache_stats(
                        layer_idx, already_there, need_to_load,
                        needed, on_device)

                reusable_slots = [
                    slot for slot, eid in slot_owner.items()
                    if eid not in needed
                ]
                victims = None
                if self.cache_policy is not None:
                    victims = iter(self.cache_policy.choose_victims(
                        layer_idx,
                        slot_owner,
                        protected=needed,
                        count=len(ordered_misses),
                    ))

                swaps = []
                used_slots = set()
                for incoming_eid in ordered_misses:
                    cpu_slot = self._eid_to_cpu_slot[
                        layer_idx][incoming_eid]
                    if cpu_slot < 0:
                        raise RuntimeError(
                            "Exclusive ownership expected incoming expert on "
                            f"CPU: layer={layer_idx}, expert={incoming_eid}")
                    if victims is not None:
                        victim_eid = next(victims, None)
                        slot = (-1 if victim_eid is None else
                                self._eid_to_npu_slot[layer_idx][victim_eid])
                    elif reusable_slots:
                        slot = reusable_slots.pop()
                        victim_eid = slot_owner[slot]
                    else:
                        slot = -1
                        victim_eid = None
                    if slot < 0 or slot in used_slots:
                        if is_prefetch:
                            break
                        raise RuntimeError(
                            "No evictable NPU expert slot for reactive "
                            f"exclusive swap: layer={layer_idx}, "
                            f"needed={sorted(needed)}, "
                            f"resident={sorted(on_device)}")
                    used_slots.add(slot)
                    swaps.append((slot, incoming_eid,
                                  int(victim_eid), cpu_slot))

                if self._debug:
                    flag = ("[PREFETCH-SWAP]" if is_prefetch
                            else "[UPDATE-SWAP]")
                    already_there_layer = (
                        set(topk_ids_h[0].tolist()) & on_device)
                    requests = len(already_there) + len(need_to_load)
                    hit_rate = (len(already_there) / requests
                                if requests else 0.0)
                    logger.info(
                        "%s l=%d expert_hit=%s expert_miss=%s "
                        "hit_rate=%.2f layer_expert_hit=%s needed=%s "
                        "topk_ids_h=%s swaps=%s",
                        flag, layer_idx, sorted(already_there),
                        sorted(need_to_load), hit_rate,
                        already_there_layer, needed, topk_ids_h,
                        [(slot, incoming, victim)
                         for slot, incoming, victim, _ in swaps],
                    )
                    if len(swaps) < len(ordered_misses):
                        logger.info(
                            "%s l=%d SHORTFALL: need %d load but only %d "
                            "swaps planned, to_load=%s",
                            flag, layer_idx, len(ordered_misses), len(swaps),
                            ordered_misses[len(swaps):][:20],
                        )
                self._swap_expert_weights(
                    layer, layer_idx, swaps, log2phy_np)

    def _update_weights_replicated(
        self,
        topk_ids_h,
        log2phy_np,
        layer,
        layer_idx: int,
        topk_weights_h,
        is_prefetch: bool,
    ) -> None:
        """Single-card decode/prefetch planner for replicated CPU storage."""
        with torch_npu.npu.stream(self.load_stream):
            # Hotness observation only on the reactive (non-prefetch) H2D path
            # with LRC policy enabled.
            if not is_prefetch and self.cache_policy is not None:
                router_scores = topk_weights_h.tolist() if topk_weights_h is not None else None
                needed = self.cache_policy.observe(
                    layer_idx,
                    topk_ids_h.tolist(),
                    router_scores=router_scores,
                )
            else:
                needed = set(topk_ids_h.reshape(-1).tolist())

            l2p_list = log2phy_np.tolist()
            slot_owner = {s: e for e, s in enumerate(l2p_list) if s >= 0}
            on_device = set(slot_owner.values())

            if is_prefetch:
                # Prefetch: only load the truly-missing top-N predicted experts.
                ordered_misses = [e for e in topk_ids_h.reshape(-1).tolist() if e not in on_device]
                need_to_load = set(ordered_misses[:self.prefetch_topk])
            else:
                need_to_load = needed - on_device
            already_there = needed & on_device              # for cache_stats / debug

            if self.cache_policy is not None:
                self._record_cache_stats(layer_idx, already_there, need_to_load, needed, on_device)
            reusable_slots = [s for s, e in slot_owner.items()
                            if e not in needed]          # slots to recycle

            if self._debug:
                flag = '[PREFETCH-W]' if is_prefetch else '[UPDATE-W]'
                already_there_layer = set(topk_ids_h[0].tolist()) & on_device
                logger.info("%s l=%d expert_hit=%s expert_miss=%s hit_rate=%.2f layer_expert_hit=%s needed=%s topk_ids_h=%s" ,
                            flag,layer_idx, sorted(already_there),
                            # sorted(need_to_load), len(already_there_layer) / topk_ids_h.shape[1],
                            sorted(need_to_load), len(already_there) / (len(already_there) + len(need_to_load)) if len(already_there) + len(need_to_load) > 0 else 0,
                            already_there_layer, needed, topk_ids_h)
                if need_to_load and len(need_to_load) > len(reusable_slots):
                    logger.info("%s l=%d SHORTFALL: need %d load but only %d slots, "
                                "to_load=%s",
                                flag,layer_idx, len(need_to_load), len(reusable_slots),
                                sorted(need_to_load)[:20])

            n_copies = 0
            planned_loads = []
            victims = None
            if self.cache_policy is not None:
                victims = iter(self.cache_policy.choose_victims(
                    layer_idx,
                    slot_owner,
                    protected=needed,
                    count=len(need_to_load),
                ))
            for eid in need_to_load:
                if self.cache_policy is not None:
                    victim = next(victims, None)
                    slot = int(log2phy_np[victim]) if victim is not None else -1
                elif reusable_slots:
                    slot = reusable_slots.pop()
                    victim = slot_owner[slot]
                else:
                    slot = -1
                    victim = None

                if slot < 0:
                    if self._debug:
                        logger.info(
                            "[UPDATE-W] l=%d NO SLOTS: %d experts could not be loaded, "
                            "missed=%s",
                            layer_idx, len(need_to_load) - n_copies,
                            sorted(list(need_to_load))[n_copies:][:20])
                    break  # no free slots — should not happen in normal usage
                
                planned_loads.append((slot, eid))
                # Update mapping
                if victim is None:
                    victim = slot_owner[slot]
                log2phy_np[victim] = -1             # evict old occupant
                on_device.discard(victim)
                log2phy_np[eid] = slot               # assign slot to new expert
                slot_owner[slot] = eid
                on_device.add(eid)
                if slot in reusable_slots:
                    reusable_slots.remove(slot)
                n_copies += 1

            self._load_expert_weights_into_slots(
                layer, layer_idx, planned_loads)
            self._synchronize_h2d()

    def _update_weights(self, args):
        """Apply common routing updates, then dispatch by storage strategy."""
        if len(args) == 6:
            (topk_ids_h, log2phy_np, layer, layer_idx, topk_weights_h,
             is_prefetch) = args
            do_substitution = False
            router_logits_h = None
            scoring_func = "softmax"
            correction_bias_h = None
        else:
            (topk_ids_h, log2phy_np, layer, layer_idx, topk_weights_h,
             is_prefetch, do_substitution, router_logits_h, scoring_func,
             correction_bias_h) = args
        if do_substitution:
            original_ids = (
                topk_ids_h[:, :self.topk].clone()
                if self._debug else None
            )
            substituted_ids = substitute_experts(
                router_logits_h,
                topk_ids_h[:, :self.topk],
                torch.from_numpy(log2phy_np),
                expert_substitution_threshold=(
                    self.offload_config.expert_substitution_threshold),
                scoring_func=scoring_func,
                e_score_correction_bias=correction_bias_h,
            )
            if original_ids is not None:
                self._log_expert_substitution(
                    layer_idx, original_ids, substituted_ids)
            topk_ids_h[:, :self.topk].copy_(substituted_ids)

        update_strategy = (
            self._update_weights_exclusive_dynamic
            if self.exclusive_dynamic_enabled
            else self._update_weights_replicated
        )
        update_strategy(
            topk_ids_h,
            log2phy_np,
            layer,
            layer_idx,
            topk_weights_h,
            is_prefetch,
        )

    def _seed_single_card_hot_experts(self) -> None:
        """Seed cache state for experts already placed during weight load."""
        for layer_idx in range(len(self.moe_layers)):
            pairs = self.offload_config.hot_expert_pairs_for_layer(
                layer_idx, self.num_total_experts)
            if not pairs:
                logger.warning("[HOT-PRELOAD] l=%d missing in json; "
                               "using ordinary initial placement", layer_idx)
                continue
            ndev = self.num_device_experts_for_layer(layer_idx)
            weights = dict(pairs[:ndev])
            if self.exclusive_dynamic_enabled:
                expected = (
                    self.offload_config.initial_device_experts_for_layer(
                        layer_idx, self.num_total_experts))
                actual = self._npu_slot_to_eid[layer_idx]
                if actual != expected:
                    raise RuntimeError(
                        "Exclusive hot experts were not placed during weight "
                        f"initialization: layer={layer_idx}, "
                        f"expected={expected}, actual={actual}")
                self._validate_exclusive_ownership(layer_idx)
            if self.cache_policy is not None:
                self.cache_policy.seed_layer_hotness(layer_idx, weights)
            if self._debug:
                logger.info(
                    "[HOT-PRELOAD] l=%d initialized %d hot experts, ids=%s",
                    layer_idx, len(weights), list(weights)[:20])

    def _preload_hot_experts(self):
        """Finalize offline hot-expert placement and seed LRC hotness.

        Triggered from _finalize_offload() when hot_expert_preload is on.
        Single-card weights are already loaded directly into their ranked
        device slots, so this path only seeds the cache. Multi-card placement
        still requires its collective rank planner and retains the historical
        post-load H2D path. No-op when the switch is off.
        """
        if not self.offload_config.hot_expert_preload:
            return
        if not self.enable_multi_card:
            self._seed_single_card_hot_experts()
            return
        if self.exclusive_shared_cpu_enabled:
            if self.cache_policy is not None:
                self._mc_lrc = self.cache_policy
            for layer_idx in range(len(self.moe_layers)):
                expected = (
                    self.offload_config.initial_device_experts_for_layer(
                        layer_idx, self.num_total_experts))
                actual = self._npu_slot_to_eid[layer_idx]
                if actual != expected:
                    raise RuntimeError(
                        "Exclusive shared hot experts were not loaded "
                        "directly into their initial global NPU slots: "
                        f"layer={layer_idx}, expected={expected}, "
                        f"actual={actual}")
                pairs = dict(
                    self.offload_config.hot_expert_pairs_for_layer(
                        layer_idx, self.num_total_experts))
                weights = {
                    eid: pairs[eid] for eid in actual if eid in pairs
                }
                if self.cache_policy is not None:
                    self.cache_policy.seed_layer_hotness(layer_idx, weights)
                self._validate_exclusive_ownership(layer_idx)
            logger.info(
                "[HOT-PRELOAD] exclusive shared experts were loaded "
                "directly during checkpoint initialization")
            return
        if self.exclusive_sharded_cpu_enabled:
            if self.cache_policy is not None:
                self._mc_lrc = self.cache_policy
            for layer_idx in range(len(self.moe_layers)):
                expected_local = (
                    self.offload_config.initial_device_experts_for_rank(
                        layer_idx, self.num_total_experts,
                        self.ep_size, self.ep_rank))
                if self._npu_slot_to_eid[layer_idx] != expected_local:
                    raise RuntimeError(
                        "Rank-local exclusive hot experts were not loaded "
                        "directly into their initial NPU slots: "
                        f"rank={self.ep_rank}, layer={layer_idx}, "
                        f"expected={expected_local}, "
                        f"actual={self._npu_slot_to_eid[layer_idx]}")
                all_initial = []
                for rank in range(self.ep_size):
                    all_initial.extend(
                        self.offload_config.initial_device_experts_for_rank(
                            layer_idx, self.num_total_experts,
                            self.ep_size, rank))
                pairs = dict(
                    self.offload_config.hot_expert_pairs_for_layer(
                        layer_idx, self.num_total_experts))
                weights = {
                    eid: pairs[eid] for eid in all_initial if eid in pairs
                }
                if self.cache_policy is not None:
                    self.cache_policy.seed_layer_hotness(layer_idx, weights)
                self._validate_exclusive_ownership(layer_idx)
            logger.info(
                "[HOT-PRELOAD] rank-local exclusive experts were loaded "
                "directly during checkpoint initialization")
            return
        if self.cache_policy is not None:
            # Multi-card decode uses _mc_lrc; share the already configured
            # per-layer policy so offline seeds and runtime observations evolve
            # from the same state.
            self._mc_lrc = self.cache_policy
        with torch_npu.npu.stream(self.load_stream):
            for layer_idx, layer in enumerate(self.moe_layers):
                pairs = self.offload_config.hot_expert_pairs_for_layer(
                    layer_idx, self.num_total_experts)
                if not pairs:
                    logger.warning("[HOT-PRELOAD] l=%d missing in json, skip",
                                   layer_idx)
                    continue
                from vllm_ascend.expert_offload.multi_card_planner import (
                    plan_hot_preload)
                per_rank_slots = (
                    self.offload_config.num_device_experts_for_rank(
                        layer_idx, self.ep_size))
                force_shard = (
                    getattr(self, '_shard_size', None)
                    if (self.offload_config.shard_per_rank
                        and not self._shared_h2d_layer_ready(layer_idx))
                    else None)
                placement = plan_hot_preload(
                    pairs,
                    global_num_experts=self.num_total_experts,
                    ep_size=self.ep_size,
                    num_device_experts=per_rank_slots,
                    force_shard=force_shard,
                )
                my_experts = placement.per_rank_experts[self.ep_rank]
                for slot, eid in enumerate(my_experts):
                    if eid >= 0:
                        self._load_expert_weights_into_slot(
                            layer, layer_idx, eid, slot)
                self.log2phy_h.copy_(placement.log2phy)
                self._mc_prev_log2phy[layer_idx] = placement.log2phy.clone()
                self._mc_resident[layer_idx] = {
                    slot: int(eid)
                    for slot, eid in enumerate(my_experts) if eid >= 0
                }
                pair_weights = dict(pairs)
                weights = {
                    eid: pair_weights[eid]
                    for eid, physical_id in enumerate(
                        placement.log2phy.tolist())
                    if physical_id >= 0
                }
                layer.log2phy.copy_(self.log2phy_h)         # H2D writeback
                if self.torch_shared_cpu_weights_enabled:
                    # Release pageable->pinned staging after each layer so
                    # offline hot preload cannot retain a model-wide pinned
                    # duplicate until the final synchronization.
                    self._synchronize_h2d()
                if self.cache_policy is not None:
                    self.cache_policy.seed_layer_hotness(layer_idx, weights)
                if self._debug:
                    logger.info(
                        "[HOT-PRELOAD] l=%d loaded %d hot experts, ids=%s",
                        layer_idx, len(weights), list(weights)[:20])
            self._synchronize_h2d()

    def predict_next_layer_experts_npu(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Predict which experts layer layer_idx+1 will need, on NPU.

        Runs entirely on the NPU so it can be captured in a CUDA/NPU graph.
        The returned tensors live on NPU.

        Two paths:
        - Hash-routed layers (first num_hash_layers): experts come from the
          tid2eid table indexed by token id (deterministic, 100% accurate).
        - Learned layers (the rest): softmax + topk on the gate logits of the
          first token.

        Args:
            layer_idx: Current layer index.
            hidden_states: [num_tokens, hidden_dim] NPU tensor.

        Returns:
            (topk_weights, topk_ids) for the first token only, both
            [1, topk] NPU tensors, or None if prediction is not possible.
        """
        next_idx = layer_idx + 1
        if next_idx >= len(self.moe_layers):
            return None  # last layer — nothing to prefetch

        if next_idx >= len(self._gate_weights_npu):
            return None
        gate_w = self._gate_weights_npu[next_idx]
        if gate_w is None:
            return None

        # Hash-routed layers (the first num_hash_layers in DeepSeek-V4):
        # experts are a deterministic function of the token id via the tid2eid
        # table, NOT of the gate logits. The learned softmax+topk path below
        # would predict the *wrong* experts for these layers, so look the
        # table up directly — perfect prediction, no matmul needed.
        # See md_analysis/2026-0711-1430-ds-v4前三层的topk计算规则.md §7.
        next_layer = self.moe_layers[next_idx]
        tid2eid = getattr(getattr(next_layer, "gate", None), "tid2eid", None)
        if tid2eid is not None:
            input_ids = get_forward_context().input_ids
            if input_ids is None:
                return None
            # [1] on the tid2eid device -> index_select -> [1, topk] int32.
            first_id = input_ids[:1].to(tid2eid.device).long()
            topk_ids = tid2eid.index_select(0, first_id)
            # Selection does not depend on affinity for hash layers, so the
            # router weight is meaningless; uniform placeholders keep the
            # [1, topk] shape the caller expects (cache_router_weight is
            # optional).
            topk_weights = torch.full(
                (1, self.topk), 1.0 / self.topk,
                dtype=torch.float32, device=topk_ids.device,
            )
            return topk_weights, topk_ids

        # Predict the first token's full top-k candidate list. The H2D path
        # independently caps transfers with prefetch_topk, so if a higher-ranked
        # candidate is already resident it can still prefetch the next miss.
        # Keeping the full width also matches the fixed-width CPU staging buffer.
        # On-device prediction: [1, hidden_dim] x [n_experts, hidden_dim]^T
        router_logits = F.linear(hidden_states[:1].float(), gate_w)
        probs = router_logits.softmax(dim=-1)
        topk_weights, topk_ids = probs.topk(self.topk, dim=-1)
        return topk_weights, topk_ids

    def trigger_next_layer_prefetch(self, layer,
                        hidden_states: torch.Tensor | None = None) -> int:
        """Trigger next-layer expert prefetch after the GMM kernel submits.

        Graph-compatible (mirrors the reactive update_weights path — NO stream
        switch, which would break NPU capture_end): record ready_to_load_event
        on the compute stream, then _launch_host_func registers the prefetch as
        a host callback (re-run every replay). The callback runs the planner+H2D
        inner (load_stream inside gives overlap with subsequent compute) and
        records load_done_event on load_stream for the next layer's reactive to
        stream-join. Eager mode keeps the prefetch-stream overlap path.
        """
        if not self.offload_config.expert_prefetch_enabled:
            return
        if self._skip_prefill:
            return
        try:
            layer_idx = self.moe_layers.index(layer)
        except ValueError:
            return

        staged = self._stage_predicted_topk(layer_idx, hidden_states)
        if staged is None:
            return
        topk_ids_h, topk_weights_h, log2phy_h, log2phy_np, next_layer, next_idx = staged

        ready_to_load_event = torch_npu.npu.Event()
        torch_npu.npu.current_stream().record_event(ready_to_load_event) 
        with torch_npu.npu.stream(self._prefetch_stream):
            self._prefetch_stream.wait_event(ready_to_load_event)
            current_compute_stream = torch_npu.npu.current_stream()
            subscribed_compute_streams = get_subscribed_compute_streams()
            if current_compute_stream not in subscribed_compute_streams:
                torch_npu.npu._subscribe_report(current_compute_stream)
                subscribed_compute_streams.add(current_compute_stream)

            prefetch_fn, prefetch_args = self._build_prefetch_call(
                topk_ids_h, topk_weights_h, log2phy_h, log2phy_np, next_layer, next_idx)
            nxt = next_idx

            def _prefetch_host_cb(_args):
                prefetch_fn(_args)
                

            if _EXTRA_CTX.capturing:
                if self.enable_multi_card:
                    self._log_mc_debug_schedule(
                        next_idx, is_prefetch=True)
                torch_npu.npu._launch_host_func(
                    current_compute_stream, _prefetch_host_cb, prefetch_args)
            else:
                _prefetch_host_cb(prefetch_args)

            next_layer.log2phy.copy_(log2phy_h, non_blocking=_EXTRA_CTX.capturing)
            # 记录一个传输流完成的事件，用于后续主流和它汇聚
            load_done_event = torch_npu.npu.Event()
            self._prefetch_stream.record_event(load_done_event)
            with self._prefetch_state_lock:
                self._prefetch_layer_npu_event[nxt] = load_done_event

    def _stage_predicted_topk(self, layer_idx, hidden_states):
        """Resolve the next layer, predict its experts on-device, and D2H-stage
        them into pinned buffers. Returns (topk_ids_h, topk_weights_h, log2phy_h,
        log2phy_np, next_layer, next_idx), or None if prefetch isn't possible
        (last layer / missing gate weights / prediction failed)."""
        next_idx = layer_idx + 1
        if next_idx >= len(self.moe_layers) - 1:
            return None
        predicted = self.predict_next_layer_experts_npu(layer_idx, hidden_states)
        if predicted is None:
            return None
        topk_weights, topk_ids = predicted
        next_layer = self.moe_layers[next_idx]
        num_tokens = topk_ids.size(0)
        topk_ids_h = self.topk_ids_h[:num_tokens]
        topk_ids_h.copy_(topk_ids.to(torch.int32), non_blocking=_EXTRA_CTX.capturing)
        topk_weights_h = None
        if (self.cache_policy is not None and topk_weights is not None
                and self.offload_config.cache_router_weight != 0):
            topk_weights_h = self.topk_weights_h[:num_tokens]
            topk_weights_h.copy_(topk_weights.to(dtype=torch.float32),
                                 non_blocking=_EXTRA_CTX.capturing)
        log2phy_h = self._prefetch_log2phy_h
        log2phy_h.copy_(next_layer.log2phy, non_blocking=_EXTRA_CTX.capturing)
        return (topk_ids_h, topk_weights_h, log2phy_h, self._prefetch_log2phy_np,
                next_layer, next_idx)

    def _build_prefetch_call(self, topk_ids_h, topk_weights_h, log2phy_h,
                             log2phy_np, next_layer, next_idx):
        """Pick the single-card vs multi-card planner+H2D inner and its args."""
        self._is_prefetch = True
        if self.enable_multi_card:
            per_rank_slots = self.offload_config.num_device_experts_for_rank(
                next_idx, self.ep_size)
            # mc2_mask_h=None: prefetch predicts the NEXT layer whose
            # active-token mask isn't known yet, so don't filter. Prefetch
            # placement is corrected by the next layer's reactive update
            # (which does filter), and prefetch calls are excluded from
            # hit-rate stats via is_prefetch=True.
            args = (
                topk_ids_h, log2phy_h, next_layer, next_idx, per_rank_slots,
                True, None)
            if getattr(self, "_debug", False):
                args += (_EXTRA_CTX.capturing,)
            return self._update_weights_multi_card, args
        return self._update_weights, (
            topk_ids_h, log2phy_np, next_layer, next_idx, topk_weights_h,
            self._is_prefetch)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _record_cache_stats(
        self,
        layer_idx: int,
        hit_experts: set[int],
        miss_experts: set[int],
        needed: set[int],
        on_device: set[int],
    ):
        self.cache_calls[layer_idx] += 1
        self.cache_requests[layer_idx] += len(needed)
        self.cache_hits[layer_idx] += len(hit_experts)
        self.cache_misses[layer_idx] += len(miss_experts)
        self.last_hit_experts[layer_idx] = sorted(hit_experts)
        self.last_miss_experts[layer_idx] = sorted(miss_experts)

        interval = self.offload_config.cache_stats_log_interval
        if interval == 0 or self.cache_calls[layer_idx] % interval != 0:
            return

        requests = self.cache_requests[layer_idx]
        hit_rate = self.cache_hits[layer_idx] / requests if requests else 0.0
        policy_step = -1
        if self.cache_policy is not None:
            policy_step = self.cache_policy.layer_step(layer_idx)
        logger.info(
            "[EXPERT-OFFLOAD-CACHE] layer=%d cache_step=%d calls=%d policy_step=%d "
            "hit_rate=%.4f hits=%d misses=%d last_hit=%s last_miss=%s resident=%s",
            layer_idx,
            self.cache_calls[layer_idx],
            self.cache_calls[layer_idx],
            policy_step,
            hit_rate,
            self.cache_hits[layer_idx],
            self.cache_misses[layer_idx],
            self.last_hit_experts[layer_idx],
            self.last_miss_experts[layer_idx],
            sorted(on_device),
        )



_EXPERT_OFFLOAD_MANAGER: ExpertOffloadManager = None


def maybe_init_expert_offload_manager(vllm_config: VllmConfig):
    # if no need to init offload manager:
    #     return
    global _EXPERT_OFFLOAD_MANAGER
    if _EXPERT_OFFLOAD_MANAGER is None:
        _EXPERT_OFFLOAD_MANAGER = ExpertOffloadManager(vllm_config)


def has_expert_offload_manager():
    return _EXPERT_OFFLOAD_MANAGER is not None


def get_expert_offload_manager():
    assert _EXPERT_OFFLOAD_MANAGER is not None, (
        "Expert Offload Manager is not initialized"
    )
    return _EXPERT_OFFLOAD_MANAGER

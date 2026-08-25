"""Ascend prefetch-based CPU offloading with NZ-format static buffers."""

from collections.abc import Generator
from dataclasses import dataclass
from functools import wraps
from typing import Any

import torch
import torch.nn as nn
import torch_npu
from vllm.logger import logger
from vllm.model_executor.offloader.base import should_pin_memory
from vllm.model_executor.offloader.prefetch import (
    ParamInfo as VllmParamInfo,
)
from vllm.model_executor.offloader.prefetch import (
    PrefetchOffloader,
    StaticBufferPool,
)
from vllm.model_executor.offloader.prefetch import (
    _CpuParamOffloader as VllmCpuParamOffloader,
)
from vllm.model_executor.offloader.prefetch import (
    _ModuleOffloader as VllmModuleOffloader,
)

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

# w4a8 (mxfp4) weights are cast with customize_dtype/input_dtype, which
# produces FRACTAL_NZ_C0_16 (50) instead of plain FRACTAL_NZ (29).
ACL_FORMAT_FRACTAL_NZ_C0_16 = 50
_NZ_FORMATS = (ACL_FORMAT_FRACTAL_NZ, ACL_FORMAT_FRACTAL_NZ_C0_16)

# mark
_ASCEND_PREFETCH_MXFP4_WEIGHT_ATTR = "_vllm_ascend_prefetch_offload_mxfp4_weight"


def mark_prefetch_offload_mxfp4_weight(param: nn.Parameter) -> None:
    """Mark a parameter whose prefetch static buffer must use the mxfp4
    format-29 (FRACTAL_NZ_C0_16) recipe.

    Used for the dual-offload full-size duplicate params (w13_weight_full /
    w2_weight_full): they are created as plain base-format tensors and
    offloaded to pinned CPU before any process_weights_after_loading runs,
    so device-side NZ detection can never see them — the mark carries the
    format requirement instead. Their CPU bytes already hold the NZ layout
    (produced by the expert-offload manager's CPU-side cast).
    """
    setattr(param, _ASCEND_PREFETCH_MXFP4_WEIGHT_ATTR, True)


def _is_prefetch_offload_mxfp4_weight(param: nn.Parameter) -> bool:
    return bool(getattr(param, _ASCEND_PREFETCH_MXFP4_WEIGHT_ATTR, False))


def _nz_format_of(param: nn.Parameter) -> int:
    """Return the NPU storage format of param if it is NZ, else 0."""
    if param.data.device.type != "npu":
        return 0

    try:
        npu_format = int(torch_npu.get_npu_format(param.data))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return 0

    return npu_format if npu_format in _NZ_FORMATS else 0


@dataclass
class ParamInfo(VllmParamInfo):
    """Ascend parameter metadata with static-buffer format requirements."""

    nz_format: int = 0


def _cast_buffer_to_nz(buffer: torch.Tensor, nz_format: int) -> torch.Tensor:
    """Cast an ND static buffer to the given NZ format, reproducing the
    exact storage layout the quant method produces for the real weight."""
    if nz_format == ACL_FORMAT_FRACTAL_NZ:
        return torch_npu.npu_format_cast(buffer, ACL_FORMAT_FRACTAL_NZ)
    if nz_format == ACL_FORMAT_FRACTAL_NZ_C0_16:
        # Mirror the w4a8 recipe in quantization/methods/fp8.py: the cast is
        # applied to the contiguous (E, N, K) view, the runtime weight is the
        # transposed (E, K, N) alias of the NZ storage.
        casted = torch_npu.npu_format_cast(
            buffer.transpose(1, 2).contiguous().view(torch.uint8),
            ACL_FORMAT_FRACTAL_NZ,
            customize_dtype=torch.float8_e4m3fn,
            input_dtype=torch_npu.float4_e2m1fn_x2,
        )
        return casted.transpose(1, 2)
    raise ValueError(f"Unsupported NZ format for prefetch static buffer: {nz_format}")


class _AscendStaticBufferPool(StaticBufferPool):
    """Static buffer pool that casts NZ buffers at allocation time.

    The base class's buffers are assigned to params (and initial prefetches
    are started) inside PrefetchOffloader.post_init(), so the NZ cast must
    happen here in __init__ -- casting afterwards would orphan the pointers
    already handed out.
    """

    def __init__(
        self,
        param_infos: list[VllmParamInfo],
        slot_capacity: int,
        device: torch.device,
    ):
        super().__init__(param_infos, slot_capacity, device)

        key_to_nz_format: dict[tuple, int] = {}
        for info in param_infos:
            nz_format = getattr(info, "nz_format", 0)
            if info.key in key_to_nz_format and key_to_nz_format[info.key] != nz_format:
                raise ValueError(
                    "Conflicting NZ buffer requirements for prefetch static buffer "
                    f"key {info.key}: parameters with different NZ formats share this key."
                )
            key_to_nz_format[info.key] = nz_format
            if nz_format:
                logger.info("%s enables NZ buffer (format %d)", info.name, nz_format)

        for key, nz_format in key_to_nz_format.items():
            if not nz_format:
                continue
            self._buffers[key] = [
                _cast_buffer_to_nz(buffer, nz_format) for buffer in self._buffers[key]
            ]


class AscendPrefetchOffloader(PrefetchOffloader):
    """Ascend prefetch offloader that reuses vLLM behavior with NZ buffers."""

    def __init__(
        self,
        group_size: int,
        num_in_group: int,
        prefetch_step: int,
        offload_params: set[str] | None = None,
        mode: str = "cpu",
    ):
        super().__init__(
            group_size=group_size,
            num_in_group=num_in_group,
            prefetch_step=prefetch_step,
            offload_params=offload_params,
            mode=mode,
        )
        self.module_offloaders: list[_ModuleOffloader] = []

    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
    ) -> list[nn.Module]:
        # MTP drafter loads its model via a second make_layers pass, which
        # would trip the upstream "wrap_modules called once" assert. Leave
        # drafter layers unwrapped: their decode-shaped forwards are served
        # by the expert-offload manager's pool/paging path.
        if self.module_offloaders:
            logger.info(
                "AscendPrefetchOffloader: modules already wrapped; leaving "
                "additional modules (e.g. MTP drafter) unwrapped")
            return list(modules_generator)

        import vllm.model_executor.offloader.prefetch as vllm_prefetch

        original_module_offloader = vllm_prefetch._ModuleOffloader
        vllm_prefetch._ModuleOffloader = _ModuleOffloader
        try:
            return super().wrap_modules(modules_generator)
        finally:
            vllm_prefetch._ModuleOffloader = original_module_offloader

    # ------------------------------------------------------------------ #
    #  Dual-offload test: prefetch serves PREFILL only; decode belongs to  #
    #  the ExpertOffloadManager.                                           #
    # ------------------------------------------------------------------ #

    def _skip_for_regime(self) -> bool:
        """Skip wait/prefetch copies outside the prefill regime.

        Only active when the expert-offload manager runs with
        prefill_module_prefetch=True; otherwise upstream behavior. The
        regime test mirrors the MoE apply() path exactly:
        multi-card: moe_comm_type != MC2 is prefill; single-card:
        num_tokens > offload_threshold is prefill. When no forward context
        (or attributes) is available, don't skip (conservative).
        """
        try:
            from vllm_ascend.expert_offload.expert_offload_manager import (
                get_expert_offload_manager, has_expert_offload_manager)
            if not has_expert_offload_manager():
                return False
            mgr = get_expert_offload_manager()
            if not mgr.offload_config.prefill_module_prefetch:
                return False
            from vllm.forward_context import get_forward_context
            ctx = get_forward_context()
            if mgr.offload_config.enable_multi_card:
                from vllm_ascend.ascend_forward_context import MoECommType
                moe_comm_type = getattr(ctx, "moe_comm_type", None)
                if moe_comm_type is None:
                    return False
                return moe_comm_type == MoECommType.MC2
            num_tokens = getattr(ctx, "num_tokens", None)
            if num_tokens is None:
                return False
            return num_tokens <= mgr.offload_threshold
        except Exception:
            return False

    def _wait_for_layer(self, layer_idx: int) -> None:
        if self._skip_for_regime():
            return
        super()._wait_for_layer(layer_idx)

    def _start_prefetch(self, layer_idx: int) -> None:
        if self._skip_for_regime():
            return
        super()._start_prefetch(layer_idx)

    def post_init(self):
        import vllm.model_executor.offloader.prefetch as vllm_prefetch

        original_buffer_pool = vllm_prefetch.StaticBufferPool
        vllm_prefetch.StaticBufferPool = _AscendStaticBufferPool
        try:
            super().post_init()
        finally:
            vllm_prefetch.StaticBufferPool = original_buffer_pool


class _AscendCpuParamOffloader(VllmCpuParamOffloader):
    """CPU param offloader that can absorb NZ-format weights.

    torch_npu's copy_ rejects any tensor with an internal (NZ) format, in
    both directions. UntypedStorage *slices*, however, shed the format tag,
    so copying through them degenerates to a raw byte memcpy (the same
    trick expert_offload uses). This lets us move NZ weights D2H/H2D as
    opaque bytes without ever changing their layout.
    """

    def __init__(self, module: nn.Module, param_name: str):
        super().__init__(module, param_name)
        self._nz_format: int = 0

    def absorb_nz_param_from_device(self, npu_format: int) -> None:
        """Move an on-device NZ param to pinned CPU storage as raw bytes.

        Called from the wrapped process_weights_after_loading, i.e. while the
        param is still on NPU in NZ format and before device_loading_context's
        finally clause runs. param.data is replaced by a CPU tensor that keeps
        the NZ tensor's logical shape/stride/dtype but holds the NZ-layout
        bytes, so device_loading_context's `.to("cpu")` becomes a no-op and
        never triggers the forbidden internal-format copy_.
        """
        param = self._param
        data = param.data
        nbytes = data.untyped_storage().nbytes()

        cpu_storage = torch.empty_strided(
            size=data.size(),
            stride=data.stride(),
            dtype=data.dtype,
            layout=data.layout,
            device="cpu",
            pin_memory=should_pin_memory(),
        )
        cpu_storage.untyped_storage().copy_(data.untyped_storage()[0:nbytes])

        self._cpu_storage = cpu_storage
        self._nz_format = npu_format
        self.offloaded_bytes = nbytes
        param.data = cpu_storage

    def copy_to_gpu_buffer(self) -> None:
        """H2D copy of CPU storage into the assigned static GPU buffer."""
        assert self._cpu_storage is not None, "CPU storage not initialized"
        assert self._gpu_buffer is not None, "GPU buffer not assigned"
        if self._nz_format:
            nbytes = self._cpu_storage.untyped_storage().nbytes()
            self._gpu_buffer.untyped_storage()[0:nbytes].copy_(
                self._cpu_storage.untyped_storage(), non_blocking=True
            )
        else:
            self._gpu_buffer.copy_(self._cpu_storage, non_blocking=True)


class _ModuleOffloader(VllmModuleOffloader):
    """vLLM module offloader with Ascend NZ-format detection."""

    def __init__(
        self,
        mode: str,
        module: nn.Module,
        copy_stream: torch.cuda.Stream,
        whitelist_param_names: list[str],
        layer_idx: int,
    ):
        import vllm.model_executor.offloader.prefetch as vllm_prefetch

        original_param_offloader = vllm_prefetch._CpuParamOffloader
        vllm_prefetch._CpuParamOffloader = _AscendCpuParamOffloader
        try:
            super().__init__(
                mode=mode,
                module=module,
                copy_stream=copy_stream,
                whitelist_param_names=whitelist_param_names,
                layer_idx=layer_idx,
            )
        finally:
            vllm_prefetch._CpuParamOffloader = original_param_offloader
        self._wrap_process_weights_for_format_detection()

    def _capture_static_buffer_formats_from_npu_params(self) -> None:
        """
        This function should be invoked during the `process_weights_after_loading`.
        And during `process_weights_after_loading`, the weights would be loaded to NPU
        for quantization and transformation of NPU formats.
        So the format of the weights could be figured out during the `process_weights_after_loading`.
        NZ params are immediately absorbed (raw-byte D2H to pinned CPU) so that
        device_loading_context never has to copy_ an internal-format tensor.

        Params marked via mark_prefetch_offload_mxfp4_weight (the dual-offload
        full-size duplicates) never sit on NPU in NZ format, so detection
        can't see them; the mark sets their NZ format directly. Their CPU
        storage already holds the NZ-layout bytes (filled by the
        expert-offload manager from its CPU-side-cast buffers).
        """
        for param_offloader in self._param_offloaders.values():
            param = param_offloader._param
            if not isinstance(param_offloader, _AscendCpuParamOffloader):
                continue
            if (_is_prefetch_offload_mxfp4_weight(param)
                    and param_offloader._nz_format == 0):
                param_offloader._nz_format = ACL_FORMAT_FRACTAL_NZ_C0_16
                continue
            nz_format = _nz_format_of(param)
            if nz_format:
                param_offloader.absorb_nz_param_from_device(nz_format)

    def _wrap_process_weights_for_format_detection(self) -> None:
        """maybe_trans_nz was called only in quantization scenario"""
        wrapped_quant_methods: set[int] = set()
        for submodule in self.module.modules():
            quant_method = getattr(submodule, "quant_method", None)
            if quant_method is not None and id(quant_method) not in wrapped_quant_methods:
                process_weights = getattr(quant_method, "process_weights_after_loading", None)
                if callable(process_weights):
                    quant_method.process_weights_after_loading = self._wrap_process_weights(process_weights)
                    wrapped_quant_methods.add(id(quant_method))

    def _wrap_process_weights(self, process_weights: Any) -> Any:
        @wraps(process_weights)
        def wrapped_process_weights(*args: Any, **kwargs: Any) -> Any:
            result = process_weights(*args, **kwargs)
            self._capture_static_buffer_formats_from_npu_params()
            return result

        return wrapped_process_weights

    def get_param_infos(self) -> list[ParamInfo]:
        infos = []
        for name, offloader in self._param_offloaders.items():
            cpu_storage = offloader._cpu_storage
            assert cpu_storage is not None, "CPU storage not initialized"
            infos.append(
                ParamInfo(
                    name=name,
                    shape=tuple(cpu_storage.shape),
                    stride=tuple(cpu_storage.stride()),
                    dtype=cpu_storage.dtype,
                    # NOTE(wangjin) different from base
                    nz_format=getattr(offloader, "_nz_format", 0),
                )
            )
        return infos

    def start_onload_to_static(self):
        """Same event-fork logic as the base class, but the H2D copies are
        delegated to the param offloaders so NZ params go through raw-byte
        storage-slice copies (copy_ rejects internal-format tensors)."""
        assert self._buffer_pool is not None, "Buffer pool not assigned"

        self._prefetch_in_capture = torch.cuda.is_current_stream_capturing()

        fork_event = torch.cuda.Event()
        torch.cuda.current_stream().record_event(fork_event)
        self.copy_stream.wait_event(fork_event)

        with torch.cuda.stream(self.copy_stream):
            for name, offloader in self._param_offloaders.items():
                cpu_storage = offloader._cpu_storage
                assert cpu_storage is not None, "CPU storage not initialized"
                assert offloader._gpu_buffer is not None, "GPU buffer not assigned"
                assert not should_pin_memory() or cpu_storage.is_pinned(), (
                    f"CPU storage for {name} is not pinned! "
                    "non_blocking=True H2D copy from non-pinned memory "
                    "causes stream synchronization that breaks "
                    "event-based fork synchronization."
                )
                if isinstance(offloader, _AscendCpuParamOffloader):
                    offloader.copy_to_gpu_buffer()
                else:
                    offloader._gpu_buffer.copy_(cpu_storage, non_blocking=True)

        self._copy_done_event.record(self.copy_stream)
        self._event_valid_for_eager = not torch.cuda.is_current_stream_capturing()

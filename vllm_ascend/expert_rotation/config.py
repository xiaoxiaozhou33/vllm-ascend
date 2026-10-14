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
"""Configuration for the static/rotating expert split (prefill-oriented offload).

Each MoE layer's local experts are divided into:
  - static experts   (first ``E - R`` local experts, resident in HBM for all layers)
  - rotating experts (last ``R`` local experts, stored in pinned host memory only;
    two shared HBM slots rotate layer blocks L / L+1 while the compute stream
    works on layer L)

The rotating count ``R`` is the knob that trades HBM capacity against H2D
bandwidth: it is intentionally NOT chosen by expert hotness (during prefill
nearly every expert is hit) and NOT fixed to half of the local experts.
"""

from __future__ import annotations

from typing import Any

from vllm.logger import logger


class ExpertRotationConfig:
    """Configuration object for ``additional_config["expert_rotation_config"]``."""

    _defaults = {
        # Master switch. When False the whole feature is dormant and the model
        # loads exactly as without this module.
        "enabled": False,
        # R: number of trailing local experts per layer that rotate between
        # host pinned memory and the two shared HBM slots.
        "num_rotating_experts": 0,
        # What to do when a forward hits an unsupported condition (graph
        # capture, draft-model-only features, the non-MegaMoe fused MC2
        # variant, ...):
        #   "sync_load" - synchronously load the rotating block on the current
        #                 stream before use (slow but numerically identical)
        #   "error"     - fail fast with a clear message
        # Note: the MegaMoe variant (enable_fused_mc2=2) IS supported — the
        # rotating experts enter the fused op as per-expert list entries.
        "fallback_mode": "sync_load",
        # When False the rotating block is loaded synchronously at the wait
        # point (phase-0 correctness mode). When True a dedicated copy stream
        # prefetches layer L+1 while layer L computes (double buffering).
        "prefetch_enabled": False,
        # Number of shared HBM rotation slots. The design fixes this at 2
        # (slot = layer_idx % 2); exposed only for experiments.
        "num_slots": 2,
        # After splitting a layer, byte-compare the absorbed pinned block plus
        # the shrunk device block against the original storage checksum.
        "verify_split": False,
        # Pre-sized allocation: create device weights directly at the STATIC
        # size S (instead of full E_local then shrink), and route the rotating
        # experts' checkpoint shards into an R-row staging param via a wrapped
        # weight_loader. Required when the full-size transient would not fit
        # (e.g. single-card EP1: 256 experts x 43 layers ~= 140GiB > card).
        "pre_sized": False,
    }

    def __init__(self, user_config: dict | None = None):
        if user_config is None:
            user_config = {}
        self.config = self._defaults.copy()
        if user_config and isinstance(user_config, dict):
            for key, value in user_config.items():
                if key in self.config:
                    self.config[key] = value
                else:
                    raise ValueError(f"expert_rotation_config has no attribute '{key}'")

        self._validate_config()

    def __getattr__(self, key: str) -> Any:
        if key in self.config:
            return self.config[key]
        raise AttributeError(f"ExpertRotationConfig has no attribute '{key}'")

    def _validate_config(self) -> None:
        if self.config["fallback_mode"] not in ("sync_load", "error"):
            raise ValueError(
                f"expert_rotation_config.fallback_mode must be 'sync_load' or 'error', "
                f"got '{self.config['fallback_mode']}'"
            )
        if self.config["num_slots"] not in (2, 4):
            raise ValueError(
                "expert_rotation_config.num_slots must be 2 (double buffering) or 4 "
                f"(deeper prefetch pipeline), got {self.config['num_slots']}"
            )
        if self.config["enabled"]:
            if not isinstance(self.config["num_rotating_experts"], int) or self.config["num_rotating_experts"] <= 0:
                raise ValueError(
                    "expert_rotation_config.num_rotating_experts must be a positive integer when enabled, "
                    f"got {self.config['num_rotating_experts']}"
                )
        logger.info(
            "[expert_rotation] enabled=%s num_rotating_experts=%s prefetch_enabled=%s fallback_mode=%s",
            self.config["enabled"],
            self.config["num_rotating_experts"],
            self.config["prefetch_enabled"],
            self.config["fallback_mode"],
        )

    def validate_against_vllm_config(self, vllm_config) -> None:
        """Cross-config validation; called from AscendConfig once vllm_config is known."""
        if not self.enabled:
            return

        problems = []
        if vllm_config.parallel_config.enable_expert_parallel is False:
            logger.warning(
                "[expert_rotation] running without expert parallelism; the split still "
                "applies to the full local expert set, but HBM savings only matter at "
                "large per-card expert counts."
            )
        if vllm_config.lora_config is not None:
            problems.append("MoE LoRA is incompatible with expert rotation")
        compilation_mode = getattr(vllm_config.compilation_config, "cudagraph_mode", None)
        if compilation_mode is not None and compilation_mode.name != "NONE":
            problems.append(
                "expert rotation requires eager execution (cudagraph_mode=NONE / --enforce-eager); "
                f"got {compilation_mode.name}"
            )
        if problems:
            raise ValueError("expert_rotation_config conflicts: " + "; ".join(problems))

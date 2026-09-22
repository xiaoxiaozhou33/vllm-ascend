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
"""Static/rotating expert split with double-buffered slot rotation (prefill-oriented).

Per MoE layer the local experts are divided into a static prefix (resident in
HBM) and a rotating suffix (pinned host memory + two shared HBM slots). See
``manager.py`` for the ownership model and ``split_apply.py`` for the MLP-stage
split.
"""

from vllm_ascend.expert_rotation.config import ExpertRotationConfig
from vllm_ascend.expert_rotation.manager import (
    ExpertRotationManager,
    get_expert_rotation_manager,
    has_expert_rotation_manager,
    maybe_init_expert_rotation_manager,
)
from vllm_ascend.expert_rotation.split_apply import rotation_split_apply

__all__ = [
    "ExpertRotationConfig",
    "ExpertRotationManager",
    "get_expert_rotation_manager",
    "has_expert_rotation_manager",
    "maybe_init_expert_rotation_manager",
    "rotation_split_apply",
]

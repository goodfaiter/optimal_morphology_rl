"""Module that owns action buffers and full per-action scaling for fixed hands."""

from __future__ import annotations

from typing import Any

import torch
from vlearn.spaces import Box
from vlearn.torch_utils.torch_jit_utils import scale

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


def _allocate_action_buffers(
    container: ModuleContainer,
    total_num_envs: int,
    device: torch.device,
) -> None:
    """Allocate action and scaled-action buffers shared with the robot."""
    action_shape = container.env.action_space.shape

    container.actions = torch.zeros((total_num_envs,) + action_shape, device=device, dtype=torch.float32)
    container.act_buf = torch.zeros_like(container.actions)
    container.last_act_buf = torch.zeros_like(container.actions)
    container.scaled_act_buf = torch.zeros_like(container.actions)


@register_module("process_fixed_actions")
class ProcessFixedActionsModule(BaseModule):
    """Owns action buffers and applies explicit per-action scales.

    Intended for fixed-base hands where the full action vector maps directly
    to motors or tendons. The config must provide ``min_action_scale`` and
    ``max_action_scale`` as lists with one entry per action dimension.

    Expects ``container.robot`` and ``container.env.action_space`` to be set
    before ``post_finalize``.
    """

    def finalize(self, container: ModuleContainer) -> None:
        """Validate dependencies."""
        if container.get("robot") is None:
            raise RuntimeError(
                "ProcessFixedActionsModule requires 'robot' in the shared container. "
                "Ensure the 'create_robot' module is listed before this module."
            )
        if container.get("env") is None:
            raise RuntimeError("ProcessFixedActionsModule requires 'env' in the shared container.")

    def post_finalize(self, container: ModuleContainer) -> None:
        """Allocate action buffers and action scaling tensors."""
        env = container.env
        if not isinstance(env.action_space, Box):
            raise RuntimeError(
                "ProcessFixedActionsModule requires env.action_space to be set. "
                "Ensure the 'robot_control' module finalize hook runs first."
            )

        _allocate_action_buffers(container, container.total_num_envs, container.device)
        self._build_action_scales(container)

    def _build_action_scales(self, container: ModuleContainer) -> None:
        """Create per-action min/max scale tensors from config."""
        device = container.device
        action_dim = container.env.action_space.shape[0]

        if "min_action_scale" not in self.config:
            raise RuntimeError("ProcessFixedActionsModule config missing 'min_action_scale'.")
        if "max_action_scale" not in self.config:
            raise RuntimeError("ProcessFixedActionsModule config missing 'max_action_scale'.")

        min_scale_cfg = self.config["min_action_scale"]
        max_scale_cfg = self.config["max_action_scale"]

        if len(min_scale_cfg) != action_dim:
            raise RuntimeError(f"min_action_scale length ({len(min_scale_cfg)}) must match action dimension ({action_dim}).")
        if len(max_scale_cfg) != action_dim:
            raise RuntimeError(f"max_action_scale length ({len(max_scale_cfg)}) must match action dimension ({action_dim}).")

        self.min_action_scale = torch.tensor(min_scale_cfg, device=device, dtype=torch.float32)
        self.max_action_scale = torch.tensor(max_scale_cfg, device=device, dtype=torch.float32)

    def step(self, container: ModuleContainer) -> None:
        """Update action history, copy new actions, and scale them."""
        container.last_act_buf[:] = container.act_buf[:]
        container.act_buf[:] = container.actions
        container.scaled_act_buf[:] = scale(
            container.act_buf,
            self.min_action_scale,
            self.max_action_scale,
        )

    def reset(self, container: ModuleContainer) -> None:
        """Zero action buffers for the environments selected by reset_buf."""
        reset_buf = container.reset_buf

        container.act_buf[reset_buf, :] = 0.0
        container.last_act_buf[reset_buf, :] = 0.0
        container.scaled_act_buf[reset_buf, :] = 0.0

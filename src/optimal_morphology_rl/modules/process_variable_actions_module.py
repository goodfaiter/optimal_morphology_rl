"""Module that owns action buffers and scales variable-base hand actions."""

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


def _build_scale_tensor(
    cfg_value: Any,
    expected_len: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """Convert a scalar or list config value to a float tensor of the expected length."""
    if isinstance(cfg_value, (list, tuple)):
        if len(cfg_value) != expected_len:
            raise RuntimeError(f"{name} length ({len(cfg_value)}) must match {expected_len}.")
        return torch.tensor(cfg_value, device=device, dtype=torch.float32)
    return torch.full((expected_len,), float(cfg_value), device=device, dtype=torch.float32)


def _build_optional_scale_tensor(
    cfg_value: Any | None,
    expected_len: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """Convert a scalar or list config value, returning an empty tensor when length is zero."""
    if expected_len == 0:
        if cfg_value is not None and cfg_value != []:
            raise RuntimeError(f"{name} should not be provided when the robot has no base velocity DOFs.")
        return torch.empty((0,), device=device, dtype=torch.float32)
    if cfg_value is None:
        raise RuntimeError(f"ProcessVariableActionsModule config missing '{name}'.")
    return _build_scale_tensor(cfg_value, expected_len, device, name)


@register_module("process_variable_actions")
class ProcessVariableActionsModule(BaseModule):
    """Owns action buffers and scales variable-base hand actions.

    For floating hands the first ``root_dim`` actions are base velocities
    scaled by ``min_velocity`` / ``max_velocity`` from config. The remaining
    actions control the active DOFs (motors or tendons) and are scaled by
    ``min_dof_scale`` / ``max_dof_scale``.

    All scale values may be scalars (broadcast to the corresponding slice) or
    lists with length matching that slice.

    Expects ``container.robot`` and ``container.env.action_space`` to be set
    before ``post_finalize``.
    """

    def finalize(self, container: ModuleContainer) -> None:
        """Validate dependencies."""
        if container.get("robot") is None:
            raise RuntimeError(
                "ProcessVariableActionsModule requires 'robot' in the shared container. "
                "Ensure the 'create_robot' module is listed before this module."
            )
        if container.get("env") is None:
            raise RuntimeError("ProcessVariableActionsModule requires 'env' in the shared container.")

    def post_finalize(self, container: ModuleContainer) -> None:
        """Allocate action buffers and scaling tensors."""
        env = container.env
        if not isinstance(env.action_space, Box):
            raise RuntimeError(
                "ProcessVariableActionsModule requires env.action_space to be set. "
                "Ensure the 'robot_control' module finalize hook runs first."
            )

        _allocate_action_buffers(container, container.total_num_envs, container.device)
        self._build_scales(container)

    def _build_scales(self, container: ModuleContainer) -> None:
        """Create velocity and active-DOF scale tensors from config."""
        device = container.device
        root_dim = container.root_slice.stop - container.root_slice.start
        num_active = container.active_dof_slice.stop - container.active_dof_slice.start

        self.min_velocity = _build_optional_scale_tensor(self.config.get("min_velocity"), root_dim, device, "min_velocity")
        self.max_velocity = _build_optional_scale_tensor(self.config.get("max_velocity"), root_dim, device, "max_velocity")

        for key in ("min_dof_scale", "max_dof_scale"):
            if key not in self.config:
                raise RuntimeError(f"ProcessVariableActionsModule config missing '{key}'.")

        self.min_dof_scale = _build_scale_tensor(self.config["min_dof_scale"], num_active, device, "min_dof_scale")
        self.max_dof_scale = _build_scale_tensor(self.config["max_dof_scale"], num_active, device, "max_dof_scale")

    def step(self, container: ModuleContainer) -> None:
        """Update action history, copy new actions, and scale them."""
        act_buf = container.act_buf
        scaled_act_buf = container.scaled_act_buf

        container.last_act_buf[:] = act_buf[:]
        act_buf[:] = container.actions

        if container.root_slice.stop > container.root_slice.start:
            scaled_act_buf[:, container.root_slice] = scale(
                act_buf[:, container.root_slice],
                self.min_velocity,
                self.max_velocity,
            )
        if container.active_dof_slice.stop > container.active_dof_slice.start:
            scaled_act_buf[:, container.active_dof_slice] = scale(
                act_buf[:, container.active_dof_slice],
                self.min_dof_scale,
                self.max_dof_scale,
            )

    def reset(self, container: ModuleContainer) -> None:
        """Zero action buffers for the environments selected by reset_buf."""
        reset_buf = container.reset_buf

        container.act_buf[reset_buf, :] = 0.0
        container.last_act_buf[reset_buf, :] = 0.0
        container.scaled_act_buf[reset_buf, :] = 0.0

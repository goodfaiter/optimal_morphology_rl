"""Module that owns action buffers, action history, and action scaling."""

from __future__ import annotations

from typing import Any

import torch
from vlearn.spaces import Box
from vlearn.torch_utils.torch_jit_utils import scale

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.create_robot_module import Robot
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

    container.robot.scaled_act_buf = container.scaled_act_buf


def _scale_actions(robot: Robot, act_buf: torch.Tensor, scaled_act_buf: torch.Tensor) -> None:
    """Scale raw actions into the robot's control ranges."""
    if robot.root_slice.stop > robot.root_slice.start:
        scaled_act_buf[:, robot.root_slice] = scale(
            act_buf[:, robot.root_slice],
            -robot.velocity_scale[robot.root_slice],
            robot.velocity_scale[robot.root_slice],
        )
    if robot.dof_slice.stop > robot.dof_slice.start:
        scaled_act_buf[:, robot.dof_slice] = scale(
            act_buf[:, robot.dof_slice],
            robot.min_revolute_scale,
            robot.max_revolute_scale,
        )


@register_module("process_actions")
class ProcessActionsModule(BaseModule):
    """Owns action buffers, updates action history, and scales actions.

    Expects ``container.robot`` to be populated by the ``create_robot`` module
    and ``container.env.action_space`` to be set by ``robot_control`` during
    ``finalize``.

    This module must be listed before ``robot_control`` in
    ``pre_physics_step_modules`` so that ``robot_control`` receives a fully
    populated ``scaled_act_buf``.
    """

    def finalize(self, container: ModuleContainer) -> None:
        """Validate dependencies that are already populated."""
        if container.get("robot") is None:
            raise RuntimeError(
                "ProcessActionsModule requires 'robot' in the shared container. "
                "Ensure the 'create_robot' module is listed before 'process_actions'."
            )
        if container.get("env") is None:
            raise RuntimeError("ProcessActionsModule requires 'env' in the shared container.")

    def post_finalize(self, container: ModuleContainer) -> None:
        """Allocate action buffers and attach scaled buffer to the robot."""
        env = container.env
        if not isinstance(env.action_space, Box):
            raise RuntimeError(
                "ProcessActionsModule requires env.action_space to be set. "
                "Ensure the 'robot_control' module is listed before or alongside "
                "'process_actions' so that its finalize hook runs first."
            )

        _allocate_action_buffers(container, container.total_num_envs, container.device)

    def step(self, container: ModuleContainer) -> None:
        """Update action history, copy new actions, and scale them."""
        container.last_act_buf[:] = container.act_buf[:]
        container.act_buf[:] = container.actions
        _scale_actions(container.robot, container.act_buf, container.scaled_act_buf)

    def reset(self, container: ModuleContainer) -> None:
        """Zero action buffers for the environments selected by reset_buf."""
        reset_buf = container.reset_buf

        container.act_buf[reset_buf, :] = 0.0
        container.last_act_buf[reset_buf, :] = 0.0
        container.scaled_act_buf[reset_buf, :] = 0.0

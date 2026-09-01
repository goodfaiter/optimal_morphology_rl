"""Module that handles action scaling, robot buffer allocation, and control."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from vlearn.spaces import Box

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("robot_control")
class RobotControlModule(BaseModule):
    """Owns robot action scaling, buffer allocation, and per-step control.

    Expects ``container.robot`` to be populated by the ``create_robot`` module.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

    def finalize(self, container: ModuleContainer) -> None:
        """Set the environment action space from the robot DOFs."""
        if container.get("robot") is None:
            raise RuntimeError(
                "RobotControlModule requires 'robot' in the shared container. "
                "Ensure the 'create_robot' module is listed before 'robot_control'."
            )

        env = container.env
        robot = container.robot
        num_actions = robot.get_num_actions()

        env.action_space = Box(
            low=np.full(num_actions, -1.0, dtype=np.float32),
            high=np.full(num_actions, 1.0, dtype=np.float32),
            dtype=np.float32,
        )

    def post_finalize(self, container: ModuleContainer) -> None:
        """Allocate action buffers, robot buffers, and create GPU commands."""
        env = container.env
        robot = container.robot
        total_num_envs = container.total_num_envs
        device = container.device

        container.act_buf = torch.zeros(
            (total_num_envs,) + env.action_space.shape,
            device=device,
            dtype=torch.float32,
        )

        robot.allocate_buffers(total_num_envs, device)

        env.inverse_reset_buf = torch.zeros(
            total_num_envs, device=device, dtype=torch.bool
        )
        env.last_act_buf = torch.zeros_like(env.act_buf)
        env.scaled_act_buf = torch.zeros_like(env.act_buf)

        robot.create_gpu_commands(
            container.env_group, container.gym, container.reset_buf, env.inverse_reset_buf
        )

    def step(self, container: ModuleContainer) -> None:
        """Apply wrist velocity, joint motor commands, and gravity compensation."""
        env = container.env
        robot = container.robot
        env.last_act_buf[:] = env.act_buf[:]
        robot.pre_physics_step(container.gym, env.act_buf)

    def reset(self, container: ModuleContainer) -> None:
        """Reset robot state for the environments selected by the reset buffer."""
        env = container.env
        robot = container.robot
        reset_config = container.get("robot_reset_config", {})

        env.act_buf[container.reset_buf, :] = 0.0
        env.last_act_buf[container.reset_buf, :] = 0.0

        robot.reset_idx(
            container.gym,
            container.reset_buf,
            container.device,
            fric_coeff=reset_config.get("fric_coeff", None),
            randomize_pose=reset_config.get("randomize_pose", False),
        )

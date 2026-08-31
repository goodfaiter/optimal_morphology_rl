"""Module that handles action scaling, robot buffer allocation, and control."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from vlearn.spaces import Box

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("robot_control")
class RobotControlModule(BaseModule):
    """Owns robot action scaling, buffer allocation, and per-step control.

    Expects ``container.robot`` to be populated by the ``robot`` module.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

    def finalize(self, env: Any) -> None:
        container = env.module_manager.container
        if container.get("robot") is None:
            raise RuntimeError(
                "RobotControlModule requires 'robot' in the shared container. "
                "Ensure the 'robot' module is listed before 'robot_control'."
            )

        robot = container.robot
        num_actions = robot.get_num_actions()

        env.action_space = Box(
            low=np.full(num_actions, -1.0, dtype=np.float32),
            high=np.full(num_actions, 1.0, dtype=np.float32),
            dtype=np.float32,
        )

    def post_finalize(self, env: Any) -> None:
        container = env.module_manager.container
        robot = container.robot
        total_num_envs = env.total_num_envs
        device = env.device

        robot.allocate_buffers(total_num_envs, device)

        env.inverse_reset_buf = torch.zeros(
            total_num_envs, device=device, dtype=torch.bool
        )
        env.last_act_buf = torch.zeros_like(env.act_buf)
        env.scaled_act_buf = torch.zeros_like(env.act_buf)

        robot.create_gpu_commands(
            container.env_group, container.gym, env.reset_buf, env.inverse_reset_buf
        )

    def pre_physics_step(self, env: Any) -> None:
        robot = env.module_manager.container.robot
        env.last_act_buf[:] = env.act_buf[:]
        robot.pre_physics_step(env.module_manager.container.gym, env.act_buf)

    def reset(self, env: Any) -> None:
        container = env.module_manager.container
        robot = container.robot
        reset_config = container.get("robot_reset_config", {})

        env.act_buf[env.reset_buf, :] = 0.0
        env.last_act_buf[env.reset_buf, :] = 0.0

        robot.reset_idx(
            container.gym,
            env.reset_buf,
            env.device,
            fric_coeff=reset_config.get("fric_coeff", None),
            randomize_pose=reset_config.get("randomize_pose", False),
        )

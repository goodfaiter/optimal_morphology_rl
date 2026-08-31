"""Cube-specific orientation reward and success bonus."""

from __future__ import annotations

import math
from typing import Any

import torch

from optimal_morphology_rl.envs.hand_envs.utils import get_reward_object_name
from optimal_morphology_rl.helpers.numpy_vlearn import quaternion_to_6d
from optimal_morphology_rl.modules.rewards.reward_base_module import RewardBaseModule
from optimal_morphology_rl.modules.rewards.reward_manager_module import register_reward


@register_reward("cube_orientation_success_reward")
class CubeOrientationSuccessReward(RewardBaseModule):
    """Orientation alignment reward + large success bonus for the cube task."""

    def post_finalize(self, env: Any) -> None:
        device = env.device
        total_num_envs = env.total_num_envs
        self.goal_aligned_steps_buf = torch.zeros(
            (total_num_envs,), device=device, dtype=torch.long
        )

    def reset(self, env: Any) -> None:
        self.goal_aligned_steps_buf[env.reset_buf] = 0

    def compute(self, env: Any) -> torch.Tensor | None:
        reward_object_name = get_reward_object_name(env)
        if reward_object_name != "cube":
            return None

        container = env.module_manager.container
        reward_object = container.reward_object
        quat_object_to_world = container.kinematic_sensor.quat_sensor_to_world

        _6d_object_to_world = quaternion_to_6d(quat_object_to_world)
        _6d_object_goal_to_world = quaternion_to_6d(
            reward_object.goal_quat_object_to_world
        )

        # Continuous orientation reward.
        goal_alignment = (
            torch.sum(_6d_object_goal_to_world * _6d_object_to_world, dim=-1) + 2.0
        )
        goal_alignment_normalized = goal_alignment / 4.0
        orientation_reward = torch.exp(
            -2.0 * (1.0 - goal_alignment_normalized)
        )

        scale = float(self.config.get("scale", 1.0))
        env.info["rewards"]["goal_orientation"] = (
            orientation_reward.sum().item() / env.total_num_envs
        )

        # Success bonus: keep orientation within 15 degrees for > 30 steps.
        quat_object_goal_to_world = reward_object.goal_quat_object_to_world
        quat_dot = (
            torch.sum(quat_object_goal_to_world * quat_object_to_world, dim=-1)
            .abs()
            .clamp(max=1.0)
        )
        goal_angle = 2.0 * torch.acos(quat_dot)
        is_aligned = goal_angle < math.radians(15.0)
        self.goal_aligned_steps_buf[is_aligned] += 1
        self.goal_aligned_steps_buf[~is_aligned] = 0
        goal_success = self.goal_aligned_steps_buf > 30

        success_scale = float(self.config.get("success_scale", 200.0))

        # Expose success for the termination module.
        container.goal_success = goal_success

        return scale * orientation_reward + success_scale * goal_success.float()

"""Cube-specific continuous orientation reward."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.envs.hand_envs.utils import get_reward_object_name
from optimal_morphology_rl.helpers.numpy_vlearn import quaternion_to_6d
from optimal_morphology_rl.modules.rewards.reward_base_module import RewardBaseModule
from optimal_morphology_rl.modules.rewards.reward_manager_module import register_reward


@register_reward("cube_orientation_reward")
class CubeOrientationReward(RewardBaseModule):
    """Continuous orientation alignment reward for the cube task."""

    def compute(self, env: Any) -> torch.Tensor | None:
        reward_object_name = get_reward_object_name(env)
        if reward_object_name != "cube":
            return None

        container = env.module_manager.container
        reward_object = container.reward_object
        quat_object_to_world = container.kinematic_sensor.quat_sensor_to_world

        _6d_object_to_world = quaternion_to_6d(quat_object_to_world)
        _6d_object_goal_to_world = quaternion_to_6d(reward_object.goal_quat_object_to_world)

        goal_alignment = torch.sum(_6d_object_goal_to_world * _6d_object_to_world, dim=-1) + 2.0
        goal_alignment_normalized = goal_alignment / 4.0
        orientation_reward = torch.exp(-2.0 * (1.0 - goal_alignment_normalized))

        scale = float(self.config.get("scale", 1.0))
        env.info["rewards"]["goal_orientation"] = orientation_reward.sum().item() / env.total_num_envs

        return scale * orientation_reward

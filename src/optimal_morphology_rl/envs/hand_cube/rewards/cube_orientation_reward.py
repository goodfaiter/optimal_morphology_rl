"""Cube-specific continuous orientation reward."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.envs.hand_envs.utils import get_reward_object_name
from optimal_morphology_rl.helpers.numpy_vlearn import quaternion_to_6d
from optimal_morphology_rl.modules.create_kinematic_sensor_module import (
    find_kinematic_sensor,
)
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
        sensor = find_kinematic_sensor(container, reward_object_name)
        if sensor is None:
            raise RuntimeError(
                "cube_orientation_reward: no kinematic sensor for the reward object. "
                "Ensure create_kinematic_sensor is listed."
            )
        quat_object_to_world = sensor.quat_sensor_to_world
        quat_goal_to_world = container.get("quat_goal_to_world")
        if quat_goal_to_world is None:
            raise RuntimeError(
                "cube_orientation_reward: container.quat_goal_to_world is missing. "
                "Add the goal_pose module to post_physics_step_modules before observation."
            )

        _6d_object_to_world = quaternion_to_6d(quat_object_to_world)
        _6d_object_goal_to_world = quaternion_to_6d(quat_goal_to_world)

        goal_alignment = torch.sum(_6d_object_goal_to_world * _6d_object_to_world, dim=-1) + 2.0
        goal_alignment_normalized = goal_alignment / 4.0
        orientation_reward = torch.exp(-2.0 * (1.0 - goal_alignment_normalized))

        scale = float(self.config.get("scale", 1.0))
        env.info["rewards"]["goal_orientation"] = orientation_reward.sum().item() / env.total_num_envs

        return scale * orientation_reward

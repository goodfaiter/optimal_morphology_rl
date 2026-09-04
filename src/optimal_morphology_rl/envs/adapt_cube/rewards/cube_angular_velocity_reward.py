"""Reward for angular velocity of the cube about the y-axis in world frame."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.envs.hand_envs.utils import get_reward_object_name
from optimal_morphology_rl.modules.rewards.reward_base_module import RewardBaseModule
from optimal_morphology_rl.modules.rewards.reward_manager_module import register_reward


@register_reward("cube_angular_velocity_reward")
class CubeAngularVelocityReward(RewardBaseModule):
    """Reward proportional to the y-axis angular velocity of the cube in world frame.

    The y-component is index 1 of ``angular_velocity_world`` (rad/s). The raw
    value is the absolute value, so spinning about the world y-axis in either
    direction is rewarded.
    """

    def compute(self, env: Any) -> torch.Tensor | None:
        reward_object_name = get_reward_object_name(env)
        if reward_object_name != "cube":
            return None

        container = env.module_manager.container
        angular_velocity_world = container.kinematic_sensor.angular_velocity_world
        angular_velocity_y_world = torch.sum(angular_velocity_world, dim=1)
        # angular_velocity_y_world = angular_velocity_world[:, 1]
        angular_velocity_y_world = torch.clamp(angular_velocity_y_world, None, 0.5)

        raw_reward = angular_velocity_y_world / 0.1
        scale = float(self.config.get("scale", 1.0))

        env.info["rewards"]["angular_velocity_y_world_frame"] = raw_reward.sum().item() / env.total_num_envs

        return scale * raw_reward

"""Reward for aligning the object orientation with the goal orientation."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.rewards.reward_jit_helpers import (
    _goal_orientation_reward_jit,
)
from optimal_morphology_rl.modules.rewards.reward_base_module import RewardBaseModule
from optimal_morphology_rl.modules.rewards.reward_manager_module import register_reward


@register_reward("goal_orientation_reward")
class GoalOrientationReward(RewardBaseModule):
    """Dot-product reward between object and goal 6D rotation representations."""

    def compute(self, env: Any) -> torch.Tensor:
        container = env.module_manager.container
        reward_object = container.reward_object
        quat_object_to_world = container.kinematic_sensor.quat_sensor_to_world

        scale = float(self.config.get("scale", 1.0))
        reward = _goal_orientation_reward_jit(
            quat_object_to_world,
            reward_object.goal_quat_object_to_world,
            scale,
        )

        env.info["rewards"]["goal_orientation"] = (
            reward.sum().item() / env.total_num_envs
        )
        return reward

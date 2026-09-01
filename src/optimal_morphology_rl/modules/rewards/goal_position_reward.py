"""Reward for minimizing object-to-goal position distance."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.rewards.reward_jit_helpers import (
    _goal_position_reward_jit,
)
from optimal_morphology_rl.modules.rewards.reward_base_module import RewardBaseModule
from optimal_morphology_rl.modules.rewards.reward_manager_module import register_reward


@register_reward("goal_position_reward")
class GoalPositionReward(RewardBaseModule):
    """Exponential reward based on object-to-goal distance."""

    def compute(self, env: Any) -> torch.Tensor:
        container = env.module_manager.container
        reward_object = container.reward_object
        object_pos_in_world = container.kinematic_sensor.pos_in_world

        normalize = float(self.config.get("normalize", 0.2))
        scale = float(self.config.get("scale", 1.5))

        reward = _goal_position_reward_jit(
            object_pos_in_world,
            reward_object.goal_pos_in_world,
            normalize,
            scale,
        )

        env.info["rewards"]["goal_position_reward"] = (
            reward.sum().item() / env.total_num_envs
        )
        env.info["rewards"]["goal_position_error_l2_norm_mm"] = (
            (torch.norm(object_pos_in_world - reward_object.goal_pos_in_world, dim=-1)
             .sum().item() / env.total_num_envs * 1000)
        )
        return reward

"""Reward for minimizing object-to-goal position distance."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.rewards.reward_base_module import RewardBaseModule
from optimal_morphology_rl.modules.rewards.reward_manager_module import register_reward


@register_reward("goal_position_reward")
class GoalPositionReward(RewardBaseModule):
    """Exponential reward based on object-to-goal distance."""

    def compute(self, env: Any) -> None:
        container = env.module_manager.container
        reward_object = container.reward_object
        object_pos_in_world = container.kinematic_sensor.pos_in_world

        obj_goal_dist = torch.norm(
            reward_object.goal_pos_in_world - object_pos_in_world, dim=-1
        )

        normalize = float(self.config.get("normalize", 0.2))
        obj_goal_dist_normalized = obj_goal_dist / normalize
        obj_goal_reward = torch.exp(-1.0 * obj_goal_dist_normalized**2)

        scale = float(self.config.get("scale", 1.5))
        env.info["rewards"]["goal_position_reward"] = (
            obj_goal_reward.sum().item() / env.total_num_envs
        )
        env.info["rewards"]["goal_position_error_l2_norm_mm"] = (
            obj_goal_dist.sum().item() / env.total_num_envs * 1000
        )
        env.rew_buf[:] += scale * obj_goal_reward

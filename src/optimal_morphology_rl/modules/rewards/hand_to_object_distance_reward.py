"""Reward for keeping distal links close to the object."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.rewards.reward_jit_helpers import (
    _hand_to_object_distance_reward_jit,
)
from optimal_morphology_rl.modules.rewards.reward_base_module import RewardBaseModule
from optimal_morphology_rl.modules.rewards.reward_manager_module import register_reward


@register_reward("hand_to_object_distance_reward")
class HandToObjectDistanceReward(RewardBaseModule):
    """Exponential reward based on average distal-link-to-object distance."""

    def compute(self, env: Any) -> torch.Tensor:
        container = env.module_manager.container
        robot = container.robot
        object_pos_in_world = container.kinematic_sensor.pos_in_world

        scale = float(self.config.get("scale", 0.1))
        link_positions = robot.distal_link_pos_buf.transpose(0, 1)

        reward = _hand_to_object_distance_reward_jit(
            link_positions,
            object_pos_in_world,
            scale,
        )

        env.info["rewards"]["hand_to_object_distance"] = (
            reward.sum().item() / env.total_num_envs
        )
        return reward

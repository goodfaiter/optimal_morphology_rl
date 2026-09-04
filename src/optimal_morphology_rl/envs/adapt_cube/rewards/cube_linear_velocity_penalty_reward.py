"""Penalty for linear velocity of the cube in world frame."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.envs.hand_envs.utils import get_reward_object_name
from optimal_morphology_rl.modules.rewards.reward_base_module import RewardBaseModule
from optimal_morphology_rl.modules.rewards.reward_manager_module import register_reward


@register_reward("cube_linear_velocity_penalty_reward")
class CubeLinearVelocityPenaltyReward(RewardBaseModule):
    """Penalty proportional to the magnitude of the cube's linear velocity in world frame."""

    def compute(self, env: Any) -> torch.Tensor | None:
        reward_object_name = get_reward_object_name(env)
        if reward_object_name != "cube":
            return None

        container = env.module_manager.container
        linear_velocity_world = container.kinematic_sensor.linear_velocity_world
        velocity_magnitude = torch.norm(linear_velocity_world, dim=-1)

        scale = float(self.config.get("scale", 1.0))
        raw_penalty = -velocity_magnitude

        env.info["rewards"]["linear_velocity_penalty"] = raw_penalty.sum().item() / env.total_num_envs

        return scale * raw_penalty

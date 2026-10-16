"""Reward for minimizing object-to-goal position distance."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.create_kinematic_sensor_module import (
    find_kinematic_sensor,
)
from optimal_morphology_rl.modules.rewards.reward_base_module import RewardBaseModule
from optimal_morphology_rl.modules.rewards.reward_manager_module import register_reward


@torch.jit.script
def _goal_position_reward_jit(
    object_pos: torch.Tensor,
    goal_pos: torch.Tensor,
    normalize: float,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exponential reward based on object-to-goal distance.

    Returns scaled reward and raw (unscaled) reward for logging.
    """
    dist = torch.norm(object_pos - goal_pos, dim=-1)
    raw = torch.exp(-((dist / normalize) ** 2))
    return scale * raw, raw


@register_reward("goal_position_reward")
class GoalPositionReward(RewardBaseModule):
    """Exponential reward based on object-to-goal distance."""

    def compute(self, env: Any) -> torch.Tensor:
        container = env.module_manager.container
        reward_object = container.reward_object
        sensor = find_kinematic_sensor(container, reward_object.name)
        if sensor is None:
            raise RuntimeError(
                "goal_position_reward: no kinematic sensor for the reward object. "
                "Ensure create_kinematic_sensor is listed."
            )
        object_pos_in_world = sensor.pos_in_world
        goal_pos = container.get("pos_goal_in_world")
        if goal_pos is None:
            raise RuntimeError(
                "goal_position_reward: container.pos_goal_in_world is missing. "
                "Add the goal_pose module to post_physics_step_modules before observation."
            )

        normalize = float(self.config.get("normalize", 0.2))
        scale = float(self.config.get("scale", 1.5))

        reward, raw = _goal_position_reward_jit(
            object_pos_in_world,
            goal_pos,
            normalize,
            scale,
        )

        env.info["rewards"]["goal_position_reward"] = raw.sum().item() / env.total_num_envs
        env.info["rewards"]["goal_position_error_l2_norm_mm"] = (
            torch.norm(object_pos_in_world - goal_pos, dim=-1).sum().item() / env.total_num_envs * 1000
        )
        return reward

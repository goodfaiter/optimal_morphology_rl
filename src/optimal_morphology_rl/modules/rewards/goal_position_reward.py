"""Reward for minimizing object-to-goal position distance."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.create_kinematic_sensor_module import (
    find_kinematic_sensor,
)
from optimal_morphology_rl.modules.goal_pose_module import goal_in_env_frame
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
    """Exponential reward based on object-to-goal distance, ``scale * exp(-(dist / normalize)^2)``.

    Config::

        goal_position_reward:
          scale: 1.0
          normalize: 0.02   # m
          axes: [x, y]      # optional (default [x, y, z]): the environment-frame axes the distance is measured over
    """

    _AXES = {"x": 0, "y": 1, "z": 2}

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        axes = list(self.config.get("axes", ["x", "y", "z"]))
        if not axes or any(a not in self._AXES for a in axes) or len(set(axes)) != len(axes):
            raise ValueError(f"goal_position_reward: 'axes' must be a non-empty subset of [x, y, z], got {axes}.")
        self.axes = [self._AXES[a] for a in axes]

    def compute(self, env: Any) -> torch.Tensor:
        container = env.module_manager.container
        reward_object = container.reward_object
        sensor = find_kinematic_sensor(container, reward_object.name)
        if sensor is None:
            raise RuntimeError(
                "goal_position_reward: no kinematic sensor for the reward object. "
                "Ensure create_kinematic_sensor is listed."
            )
        object_pos_in_world = sensor.pos_in_world[:, self.axes]
        goal_pos, _ = goal_in_env_frame(container)
        goal_pos = goal_pos[:, self.axes]

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

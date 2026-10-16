"""Reward for aligning the object orientation with the goal orientation."""

from __future__ import annotations

from typing import Any

import torch
from vlearn.torch_utils.torch_jit_utils import quat_rotate

from optimal_morphology_rl.modules.create_kinematic_sensor_module import (
    find_kinematic_sensor,
)
from optimal_morphology_rl.modules.rewards.reward_base_module import RewardBaseModule
from optimal_morphology_rl.modules.rewards.reward_manager_module import register_reward


@torch.jit.script
def _quaternion_to_6d_jit(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (x, y, z, w) to 6D rotation representation."""
    num_envs = q.shape[0]
    device = q.device
    e1 = torch.tensor([1.0, 0.0, 0.0], device=device).unsqueeze(0).expand(num_envs, 3)
    e2 = torch.tensor([0.0, 1.0, 0.0], device=device).unsqueeze(0).expand(num_envs, 3)
    col1 = quat_rotate(q, e1)
    col2 = quat_rotate(q, e2)
    return torch.cat([col1, col2], dim=-1)


@torch.jit.script
def _goal_orientation_reward_jit(
    object_quat: torch.Tensor,
    goal_quat: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dot-product reward between object and goal 6D rotation representations.

    Returns scaled reward and raw (unscaled) reward for logging.
    """
    object_6d = _quaternion_to_6d_jit(object_quat)
    goal_6d = _quaternion_to_6d_jit(goal_quat)
    raw = torch.sum(goal_6d * object_6d, dim=-1) / 2.0
    return scale * raw, raw


@register_reward("goal_orientation_reward")
class GoalOrientationReward(RewardBaseModule):
    """Dot-product reward between object and goal 6D rotation representations."""

    def compute(self, env: Any) -> torch.Tensor:
        container = env.module_manager.container
        reward_object = container.reward_object
        sensor = find_kinematic_sensor(container, reward_object.name)
        if sensor is None:
            raise RuntimeError(
                "goal_orientation_reward: no kinematic sensor for the reward object. "
                "Ensure create_kinematic_sensor is listed."
            )
        quat_object_to_world = sensor.quat_sensor_to_world
        goal_quat = container.get("quat_goal_to_world")
        if goal_quat is None:
            raise RuntimeError(
                "goal_orientation_reward: container.quat_goal_to_world is missing. "
                "Add the goal_pose module to post_physics_step_modules before observation."
            )

        scale = float(self.config.get("scale", 1.0))
        reward, raw = _goal_orientation_reward_jit(
            quat_object_to_world,
            goal_quat,
            scale,
        )

        env.info["rewards"]["goal_orientation"] = raw.sum().item() / env.total_num_envs
        return reward

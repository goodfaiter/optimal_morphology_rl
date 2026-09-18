"""Reward for minimizing fingertip-to-goal distance."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.create_kinematic_sensor_module import (
    find_kinematic_sensor,
)
from optimal_morphology_rl.modules.rewards.reward_base_module import RewardBaseModule
from optimal_morphology_rl.modules.rewards.reward_manager_module import register_reward


@torch.jit.script
def _goal_finger_tip_reward_jit(
    fingertip_pos: torch.Tensor,
    goal_pos: torch.Tensor,
    normalize: float,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exponential reward based on fingertip-to-goal distance.

    Returns scaled reward and raw (unscaled) reward for logging.
    """
    dist = torch.norm(fingertip_pos - goal_pos, dim=-1)
    raw = torch.exp(-((dist / normalize) ** 2))
    return scale * raw, raw


@register_reward("goal_finger_tip_reward")
class GoalFingerTipReward(RewardBaseModule):
    """Exponential reward based on fingertip-to-goal distance.

    Config ``link_name`` selects the fingertip kinematic sensor created by
    create_kinematic_sensor (substring match over the created sensor names,
    which are the attachment link names, e.g. ``Pinky_DP``).
    """

    def compute(self, env: Any) -> torch.Tensor:
        container = env.module_manager.container
        sensor = find_kinematic_sensor(container, str(self.config.get("link_name", "DP")))
        if sensor is None:
            raise RuntimeError(
                f"goal_finger_tip_reward: no kinematic sensor matches '{self.config.get('link_name', 'DP')}'. "
                "Ensure the .vsim defines the fingertip kinematic sensor and create_kinematic_sensor is listed."
            )
        fingertip_pos = sensor.pos_in_world
        goal_pos = container.fingertip_goal_pos

        normalize = float(self.config.get("normalize", 0.05))
        scale = float(self.config.get("scale", 1.0))

        reward, raw = _goal_finger_tip_reward_jit(fingertip_pos, goal_pos, normalize, scale)

        env.info["rewards"]["goal_finger_tip_reward"] = raw.sum().item() / env.total_num_envs
        env.info["rewards"]["goal_finger_tip_error_l2_norm_mm"] = (
            torch.norm(fingertip_pos - goal_pos, dim=-1).sum().item() / env.total_num_envs * 1000
        )
        return reward

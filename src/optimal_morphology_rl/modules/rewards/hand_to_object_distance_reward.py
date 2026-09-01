"""Reward for keeping distal links close to the object."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.rewards.reward_base_module import RewardBaseModule
from optimal_morphology_rl.modules.rewards.reward_manager_module import register_reward


@torch.jit.script
def _hand_to_object_distance_reward_jit(
    link_positions: torch.Tensor,
    object_pos: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exponential reward based on average distal-link-to-object distance.

    Returns scaled reward and raw (unscaled) reward for logging.
    """
    link_dists = torch.norm(link_positions - object_pos.unsqueeze(1), dim=-1)
    avg_dist = link_dists.mean(dim=-1)
    dist_clipped = torch.clamp(avg_dist, min=0.01)
    raw = torch.exp(-((dist_clipped / 0.2) ** 2))
    return scale * raw, raw


@register_reward("hand_to_object_distance_reward")
class HandToObjectDistanceReward(RewardBaseModule):
    """Exponential reward based on average distal-link-to-object distance."""

    def compute(self, env: Any) -> torch.Tensor:
        container = env.module_manager.container
        robot = container.robot
        object_pos_in_world = container.kinematic_sensor.pos_in_world

        scale = float(self.config.get("scale", 0.1))
        link_positions = robot.distal_link_pos_buf.transpose(0, 1)

        reward, raw = _hand_to_object_distance_reward_jit(
            link_positions,
            object_pos_in_world,
            scale,
        )

        env.info["rewards"]["hand_to_object_distance"] = raw.sum().item() / env.total_num_envs
        return reward

"""Reward for keeping distal links close to the object."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.rewards.reward_base_module import RewardBaseModule
from optimal_morphology_rl.modules.rewards.reward_manager_module import register_reward


@register_reward("hand_to_object_distance_reward")
class HandToObjectDistanceReward(RewardBaseModule):
    """Exponential reward based on average distal-link-to-object distance."""

    def compute(self, env: Any) -> None:
        container = env.module_manager.container
        robot = container.robot
        object_pos_in_world = container.kinematic_sensor.pos_in_world

        link_positions = robot.distal_link_pos_buf.transpose(0, 1)
        link_dists = torch.norm(
            link_positions - object_pos_in_world.unsqueeze(1), dim=-1
        )
        avg_dist = link_dists.mean(dim=-1)
        dist_clipped = torch.clamp(avg_dist, min=0.01)
        dist_clipped_normalized = dist_clipped / 0.2
        dist_rew = torch.exp(-1.0 * dist_clipped_normalized**2)

        scale = float(self.config.get("scale", 0.1))
        env.info["rewards"]["hand_to_object_distance"] = (
            dist_rew.sum().item() / env.total_num_envs
        )
        env.rew_buf[:] += scale * dist_rew

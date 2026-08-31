"""Penalty for large actions."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.rewards.reward_base_module import RewardBaseModule
from optimal_morphology_rl.modules.rewards.reward_manager_module import register_reward


@register_reward("action_penalty_reward")
class ActionPenaltyReward(RewardBaseModule):
    """Negative reward proportional to the squared action magnitude."""

    def compute(self, env: Any) -> None:
        action_penalty = torch.sum(env.act_buf**2, dim=-1)
        action_penalty_reward = -1 * action_penalty

        scale = float(self.config.get("scale", 0.01))
        env.info["rewards"]["action_penalty"] = (
            action_penalty_reward.sum().item() / env.total_num_envs
        )
        return scale * action_penalty_reward

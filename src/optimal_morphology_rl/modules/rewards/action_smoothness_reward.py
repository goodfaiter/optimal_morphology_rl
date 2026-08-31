"""Penalty for large action changes."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.rewards.reward_base_module import RewardBaseModule
from optimal_morphology_rl.modules.rewards.reward_manager_module import register_reward


@register_reward("action_smoothness_reward")
class ActionSmoothnessReward(RewardBaseModule):
    """Negative reward proportional to the squared action delta."""

    def compute(self, env: Any) -> None:
        action_smoothness_penalty = torch.sum(
            (env.act_buf - env.last_act_buf) ** 2, dim=-1
        )
        action_smoothness_reward = -1 * action_smoothness_penalty

        scale = float(self.config.get("scale", 0.01))
        env.info["rewards"]["action_smoothness_penalty"] = (
            action_smoothness_reward.sum().item() / env.total_num_envs
        )
        return scale * action_smoothness_reward

"""Penalty for large action changes."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.rewards.reward_jit_helpers import (
    _action_smoothness_reward_jit,
)
from optimal_morphology_rl.modules.rewards.reward_base_module import RewardBaseModule
from optimal_morphology_rl.modules.rewards.reward_manager_module import register_reward


@register_reward("action_smoothness_reward")
class ActionSmoothnessReward(RewardBaseModule):
    """Negative reward proportional to the squared action delta."""

    def compute(self, env: Any) -> torch.Tensor:
        scale = float(self.config.get("scale", 0.01))
        reward = _action_smoothness_reward_jit(
            env.act_buf,
            env.last_act_buf,
            scale,
        )

        env.info["rewards"]["action_smoothness_penalty"] = (
            reward.sum().item() / env.total_num_envs
        )
        return reward

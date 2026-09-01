"""Penalty for large actions."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.rewards.reward_base_module import RewardBaseModule
from optimal_morphology_rl.modules.rewards.reward_manager_module import register_reward


@torch.jit.script
def _action_penalty_reward_jit(
    actions: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Negative reward proportional to squared action magnitude.

    Returns scaled reward and raw (unscaled) reward for logging.
    """
    raw = -torch.sum(actions * actions, dim=-1)
    return scale * raw, raw


@register_reward("action_penalty_reward")
class ActionPenaltyReward(RewardBaseModule):
    """Negative reward proportional to the squared action magnitude."""

    def compute(self, env: Any) -> torch.Tensor:
        scale = float(self.config.get("scale", 0.01))
        reward, raw = _action_penalty_reward_jit(env.act_buf, scale)

        env.info["rewards"]["action_penalty"] = (
            raw.sum().item() / env.total_num_envs
        )
        return reward

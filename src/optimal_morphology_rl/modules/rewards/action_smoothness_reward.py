"""Penalty for large action changes."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.rewards.reward_base_module import RewardBaseModule
from optimal_morphology_rl.modules.rewards.reward_manager_module import register_reward


@torch.jit.script
def _action_smoothness_reward_jit(
    actions: torch.Tensor,
    last_actions: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Negative reward proportional to squared action delta.

    Returns scaled reward and raw (unscaled) reward for logging.
    """
    delta = actions - last_actions
    raw = -torch.sum(delta * delta, dim=-1)
    return scale * raw, raw


@register_reward("action_smoothness_reward")
class ActionSmoothnessReward(RewardBaseModule):
    """Negative reward proportional to the squared action delta."""

    def compute(self, env: Any) -> torch.Tensor:
        scale = float(self.config.get("scale", 0.01))
        reward, raw = _action_smoothness_reward_jit(
            env.act_buf,
            env.last_act_buf,
            scale,
        )

        env.info["rewards"]["action_smoothness_penalty"] = raw.sum().item() / env.total_num_envs
        return reward

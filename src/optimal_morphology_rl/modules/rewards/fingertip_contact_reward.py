"""Reward for fingertip contact with the reward object."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.rewards.reward_base_module import RewardBaseModule
from optimal_morphology_rl.modules.rewards.reward_manager_module import register_reward


@torch.jit.script
def _fingertip_contact_reward_jit(
    forces: torch.Tensor,
    contact_mask: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reward contact between monitored distal links and the reward object.

    Returns scaled reward and raw (unscaled) reward for logging.
    """
    force_against_object = forces * contact_mask
    force_against_object = torch.clamp(force_against_object, 0.0, 1.0)
    raw = torch.clamp(force_against_object.sum(dim=-1), 0.0, 3.0) / 3.0
    return scale * raw, raw


@register_reward("fingertip_contact_reward")
class FingertipContactReward(RewardBaseModule):
    """Reward contact between monitored distal links and the reward object."""

    def compute(self, env: Any) -> torch.Tensor | None:
        container = env.module_manager.container
        contacts = container.get("contacts")
        force_sensors = container.get("force_sensors")
        if contacts is None or force_sensors is None:
            return None
        if force_sensors.force_sensor_buf is None:
            return None

        forces = force_sensors.force_sensor_buf.norm(dim=-1)
        contact_mask = contacts.env_link_touch[:, contacts.monitored_link_mask]

        scale = float(self.config.get("scale", 2.0))
        reward, raw = _fingertip_contact_reward_jit(forces, contact_mask, scale)

        env.info["rewards"]["fingertip_contact_reward"] = raw.sum().item() / env.total_num_envs
        return reward

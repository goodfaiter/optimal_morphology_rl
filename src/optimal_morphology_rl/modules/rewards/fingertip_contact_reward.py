"""Reward for fingertip contact with the reward object."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.rewards.reward_base_module import RewardBaseModule
from optimal_morphology_rl.modules.rewards.reward_manager_module import register_reward


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
        force_against_object = forces * contact_mask
        force_against_object.clamp_(0.0, 1.0)
        fingertip_contact_reward = (
            torch.clamp(force_against_object.sum(dim=-1), min=0.0, max=3.0) / 3.0
        )

        scale = float(self.config.get("scale", 2.0))
        env.info["rewards"]["fingertip_contact_reward"] = (
            fingertip_contact_reward.sum().item() / env.total_num_envs
        )
        return scale * fingertip_contact_reward

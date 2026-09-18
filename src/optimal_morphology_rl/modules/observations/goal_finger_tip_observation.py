"""Observation source for a fingertip-reach goal sampled in the xz plane."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.observations.observation_base_module import (
    ObservationBaseModule,
)
from optimal_morphology_rl.modules.observations.observation_manager_module import (
    register_observation,
)


@register_observation("goal_finger_tip")
class GoalFingerTipObservation(ObservationBaseModule):
    """Fingertip-reach goal: (x, z) sampled in a configurable xz range, y fixed.

    The goal tensor is written to ``container.fingertip_goal_pos``
    ``(total_num_envs, 3)`` and observed by the policy. Goals are periodically
    resampled for random envs every step (``resample_prob`` per env per step).
    """

    def get_obs_dim(self, env: Any) -> int:
        return 3

    def _sample(self, env: Any, mask: torch.Tensor | None = None) -> None:
        """Sample the goal for the envs selected by ``mask`` (all if None)."""
        container = env.module_manager.container
        x_range, z_range = self.config.get("xz_range")
        x_min, x_max = float(x_range[0]), float(x_range[1])
        z_min, z_max = float(z_range[0]), float(z_range[1])
        y = float(self.config.get("y_fixed", 0.0))

        num = env.total_num_envs if mask is None else int(mask.sum().item())
        device = env.device
        x = x_min + (x_max - x_min) * torch.rand(num, device=device)
        z = z_min + (z_max - z_min) * torch.rand(num, device=device)
        y = torch.full((num,), y, device=device)
        goal = torch.stack([x, y, z], dim=-1)

        if mask is None:
            container.fingertip_goal_pos[:] = goal
        else:
            container.fingertip_goal_pos[mask] = goal

    def compute_observation(self, env: Any, out: torch.Tensor) -> None:
        container = env.module_manager.container
        if container.get("fingertip_goal_pos") is None:
            container.fingertip_goal_pos = torch.zeros((env.total_num_envs, 3), device=env.device, dtype=torch.float32)
            self._sample(env)

        resample_prob = float(self.config.get("resample_prob", 0.0))
        if resample_prob > 0.0:
            mask = torch.rand(env.total_num_envs, device=env.device) < resample_prob
            if mask.any():
                self._sample(env, mask)

        out[:] = container.fingertip_goal_pos

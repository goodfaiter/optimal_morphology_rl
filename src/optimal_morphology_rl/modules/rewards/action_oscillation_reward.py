"""Penalty for action oscillations in a frequency band (e.g. the few-Hz chatter of a bang-bang policy)."""

from __future__ import annotations

import math
from typing import Any

import torch

from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.rewards.reward_base_module import RewardBaseModule
from optimal_morphology_rl.modules.rewards.reward_manager_module import register_reward


@torch.jit.script
def _band_pass_jit(
    actions: torch.Tensor, last_actions: torch.Tensor, high_pass: torch.Tensor, band: torch.Tensor, a_hp: float, a_lp: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """One step of a first-order high-pass (removes slow moves) followed by a first-order low-pass (removes the
    step-to-step noise); returns the new filter states (high_pass, band)."""
    high_pass = a_hp * (high_pass + actions - last_actions)
    band = band + a_lp * (high_pass - band)
    return high_pass, band


@register_reward("action_oscillation_reward")
class ActionOscillationReward(RewardBaseModule):
    """Negative reward on the energy of the actions in the ``[low_hz, high_hz]`` band, summed over the actions.

    Each raw action is filtered per env through a first-order high-pass at ``low_hz`` (slow, intentional moves
    pass the band only weakly) and a first-order low-pass at ``high_hz`` (single-step changes at the control rate
    are damped); the penalty is ``-scale * sum(band**2)``. The filter state restarts on reset. Unlike
    'action_smoothness_reward' (the last step's change only) it sees oscillations over several steps, e.g. a
    policy switching its targets at a few Hz.

    Config shape::

        action_oscillation_reward:
          scale: 1.0
          low_hz: 2.0
          high_hz: 10.0
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.scale = float(self.config.get("scale", 1.0))
        self.low_hz = float(self.config.get("low_hz", 2.0))
        self.high_hz = float(self.config.get("high_hz", 10.0))
        if not 0.0 < self.low_hz < self.high_hz:
            raise ValueError(f"action_oscillation_reward needs 0 < low_hz < high_hz, got {self.low_hz} / {self.high_hz}.")
        self.high_pass: torch.Tensor | None = None
        self.band: torch.Tensor | None = None

    def post_finalize(self, container: ModuleContainer) -> None:
        dt = float(container.timestep) * float(container.frame_skip)
        if self.high_hz >= 0.5 / dt:
            raise ValueError(f"action_oscillation_reward high_hz {self.high_hz} must be below the Nyquist rate {0.5 / dt:g} Hz.")
        rc_hp, rc_lp = 1.0 / (2.0 * math.pi * self.low_hz), 1.0 / (2.0 * math.pi * self.high_hz)
        self.a_hp = rc_hp / (rc_hp + dt)  # discrete first-order high-pass / low-pass coefficients
        self.a_lp = dt / (rc_lp + dt)
        self.high_pass = torch.zeros_like(container.actions)
        self.band = torch.zeros_like(container.actions)

    def reset(self, container: ModuleContainer) -> None:
        reset_buf = container.get("reset_buf")
        if self.band is not None and reset_buf is not None:
            self.high_pass[reset_buf] = 0.0
            self.band[reset_buf] = 0.0

    def compute(self, env: Any) -> torch.Tensor:
        self.high_pass, self.band = _band_pass_jit(env.act_buf, env.last_act_buf, self.high_pass, self.band, self.a_hp, self.a_lp)
        raw = -torch.sum(self.band * self.band, dim=-1)
        env.info["rewards"]["action_oscillation_penalty"] = raw.sum().item() / env.total_num_envs
        return self.scale * raw

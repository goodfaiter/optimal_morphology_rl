"""Observation source for the servo torques of 'apply_m5_forces' (sim only: a privileged, critic-only observation)."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.observations.observation_base_module import (
    ObservationBaseModule,
)
from optimal_morphology_rl.modules.observations.observation_manager_module import (
    register_observation,
)

SIGNALS = ("tau_f_max", "tau_f", "tau_m", "tau_e")


@register_observation("spool_torques")
class SpoolTorquesObservation(ObservationBaseModule):
    """Servo torques of the current step per spool, from ``apply_m5_forces.last_signals`` (Nm).

    ``signals`` selects them, in order (default ``[tau_f_max]``, the friction budget); the layout is
    ``[signal 1 (num_model_tendons), signal 2 (...), ...]``. The robot cannot measure these: list the module
    under ``observation.privileged_modules`` (critic only).

    Config shape::

        spool_torques:
          signals: [tau_f_max]  # any of tau_f_max, tau_f, tau_m, tau_e
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.signals = list(self.config.get("signals", ["tau_f_max"]))
        unknown = [name for name in self.signals if name not in SIGNALS]
        if unknown or not self.signals:
            raise ValueError(f"spool_torques 'signals' must be a non-empty subset of {SIGNALS}, got {self.signals}.")

    def get_obs_dim(self, env: Any) -> int:
        spool = env.module_manager.container.get("spool")
        if spool is None:
            raise RuntimeError("spool_torques requires the 'create_spool' module (container.spool).")
        return len(self.signals) * len(spool.model_tendon_indices)

    def compute_observation(self, env: Any, out: torch.Tensor) -> None:
        last = getattr(env.module_manager.get("apply_m5_forces"), "last_signals", None)
        if last is None:  # before the first control step (the initial reset)
            out.zero_()
            return
        out[:] = torch.cat([last[name] for name in self.signals], dim=-1)

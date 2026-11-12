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

# The robot can compute tau_f_max (the friction model), tau_m (the servo law), tau_inertia (J_eff / dt * spool speed) and
# tau_e_predicted (the friction model's tau_e output, if it has one); tau_f and tau_e need the tendon load, which only
# the sim knows.
SIGNALS = ("tau_f_max", "tau_f", "tau_m", "tau_e", "tau_inertia", "tau_e_predicted")


@register_observation("spool_torques")
class SpoolTorquesObservation(ObservationBaseModule):
    """Servo torques of the current step per spool, from ``apply_m5_forces.last_signals`` (Nm).

    ``signals`` selects them, in order (default ``[tau_f_max]``, the friction budget); the layout is
    ``[signal 1 (num_model_tendons), signal 2 (...), ...]``. The robot's hand controller computes tau_f_max (the
    friction model), tau_m (the servo law), tau_inertia (J_eff / dt * omega) and tau_e_predicted (the friction model's
    prediction of tau_e: a model with a ``tau_external_Nm`` output) from its own state; tau_f and tau_e need
    the tendon load: for those, list the module under ``observation.privileged_modules`` (critic only).

    Config shape::

        spool_torques:
          signals: [tau_f_max]  # any of tau_f_max, tau_f, tau_m, tau_e, tau_inertia, tau_e_predicted
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.signals = list(self.config.get("signals", ["tau_f_max"]))
        unknown = [name for name in self.signals if name not in SIGNALS]
        if unknown or not self.signals:
            raise ValueError(f"spool_torques 'signals' must be a non-empty subset of {SIGNALS}, got {self.signals}.")

    def export_info(self, container: Any) -> dict[str, Any]:
        """``spool_torques``: the observed signals and the constants the robot's controller computes them with, per
        spool: the physics step, J_eff and the servo law's gain, K_t and i0 (nominal: a training env's spools are
        randomized, an env built in play mode is not)."""
        manager = container.env.module_manager
        spool, m5 = container.spool, manager.get("apply_m5_forces")
        num_spools = spool.j_eff.shape[-1]
        if "randomize_spool" in manager:  # the nominal J_eff
            randomize = manager.get("randomize_spool")
            j_eff = (randomize.nominal_armature + spool.slider_mass) * randomize.nominal_radius[0] ** 2
        else:
            j_eff = spool.j_eff[0]
        per_spool = lambda value: value.reshape(-1, num_spools)[0].cpu().tolist()  # noqa: E731  (the same in every env)
        return {"spool_torques": {
            "signals": list(self.signals),
            "physics_dt": float(m5.dt),
            "j_eff": j_eff.cpu().tolist(),  # kg m^2 per spool
            "motor_position_gain": per_spool(m5.motor_position_gain),  # A/rad
            "torque_constant": per_spool(m5.torque_constant),  # Nm/A
            "current_deadband": per_spool(m5.current_deadband),  # A
        }}

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
        missing = [name for name in self.signals if name not in last]
        if missing:
            raise RuntimeError(f"spool_torques: apply_m5_forces has no {missing} (tau_e_predicted needs a friction model "
                               "with a 'tau_external_Nm' output, e.g. 2026_11_10_mlp_dynamic_envelope_tau_e.pt).")
        out[:] = torch.cat([last[name] for name in self.signals], dim=-1)

"""Observation source for the servo spool state of the tendon actuator sliders."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.observations.observation_base_module import (
    ObservationBaseModule,
)
from optimal_morphology_rl.modules.observations.observation_manager_module import (
    register_observation,
)


@torch.jit.script
def _spool_state_obs_jit(slider_pos: torch.Tensor, slider_vel: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    """Spool angle theta_m = q / r and velocity omega_m = qdot / r per model tendon."""
    return torch.cat([slider_pos / radius, slider_vel / radius], dim=-1)


@register_observation("spool_state")
class SpoolStateObservation(ObservationBaseModule):
    """Spool angles and velocities of the 'create_spool' spools (what the servo encoder measures).

    For every model tendon of 'create_spool' the slider joint position and velocity are
    converted to the spool angle ``theta_m = q / r`` (rad, 0 = hand open) and velocity
    ``omega_m = qdot / r`` (rad/s), with ``r`` the spool's (per env) ``pully_radius``. The velocity is
    ``robot.joint_velocity``, i.e. the source selected by update_robot's ``velocity_source``
    (use "fd": the simulator's slider velocity misses the tendon-constrained motion).

    Layout: ``[theta_m (num_model_tendons), omega_m (num_model_tendons)]``.

    Optional observation noise (``noise``), drawn per env and tendon on every reset and constant over
    the episode; the spool itself (its command and dynamics) is unchanged:

    - ``angle_offset_range`` (rad): encoder zero offset added to ``theta_m``, so the observed 0 is
      not necessarily the true hand-open spool angle.
    - ``velocity_lag_range`` (s): time constant ``tau`` of a first-order low-pass filter on
      ``omega_m``, ``w_f += a * (w - w_f)`` with ``a = dt / (tau + dt)`` (``tau = 0``: no lag). The
      filter starts at the measured velocity on the first step after a reset.
    - ``velocity_offset_range`` (rad/s): bias added to the (filtered) ``omega_m``.

    Config shape (under ``observation.modules``)::

        spool_state:
          noise:
            angle_offset_range: [-0.1, 0.1]
            velocity_lag_range: [0.0, 0.02]
            velocity_offset_range: [-0.2, 0.2]
    """

    NOISE_KEYS = ("angle_offset_range", "velocity_lag_range", "velocity_offset_range")

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        noise = dict(self.config.get("noise") or {})
        unknown = sorted(set(noise) - set(self.NOISE_KEYS))
        if unknown:
            raise ValueError(f"spool_state 'noise' has unknown keys {unknown}; expected any of {list(self.NOISE_KEYS)}.")
        self.ranges = {key: self._range(noise, key) for key in self.NOISE_KEYS}
        if self.ranges["velocity_lag_range"][0] < 0.0:
            raise ValueError("spool_state 'noise.velocity_lag_range' must be >= 0 (seconds).")
        self.noisy = any(low != 0.0 or high != 0.0 for low, high in self.ranges.values())
        # Per env and tendon, allocated on the first reset when noise is configured.
        self.angle_offset: torch.Tensor | None = None  # rad
        self.velocity_offset: torch.Tensor | None = None  # rad/s
        self.velocity_alpha: torch.Tensor | None = None  # low-pass gain dt / (tau + dt)
        self.velocity_filtered: torch.Tensor | None = None
        self.filter_restart: torch.Tensor | None = None  # (num_envs,) start the filter at the measurement

    @staticmethod
    def _range(noise: dict[str, Any], key: str) -> tuple[float, float]:
        value = noise.get(key, [0.0, 0.0])
        if not isinstance(value, (list, tuple)) or len(value) != 2 or float(value[0]) > float(value[1]):
            raise ValueError(f"spool_state 'noise.{key}' must be a [min, max] list with min <= max, got {value}.")
        return float(value[0]), float(value[1])

    def reset(self, env: Any) -> None:
        """Draw new noise for the resetting envs."""
        if not self.noisy:
            return
        reset_buf = env.module_manager.container.get("reset_buf")
        if reset_buf is None:
            return
        spool = self._spool(env)
        if self.angle_offset is None:
            shape = (reset_buf.shape[0], len(spool.model_tendon_indices))
            self.angle_offset = torch.zeros(shape, device=reset_buf.device)
            self.velocity_offset = torch.zeros(shape, device=reset_buf.device)
            self.velocity_alpha = torch.ones(shape, device=reset_buf.device)
            self.velocity_filtered = torch.zeros(shape, device=reset_buf.device)
            self.filter_restart = torch.ones(shape[0], dtype=torch.bool, device=reset_buf.device)

        def draw(key):
            low, high = self.ranges[key]
            return (low + (high - low) * torch.rand_like(self.angle_offset))[reset_buf]

        self.angle_offset[reset_buf] = draw("angle_offset_range")
        self.velocity_offset[reset_buf] = draw("velocity_offset_range")
        dt = float(spool.dt)
        self.velocity_alpha[reset_buf] = dt / (draw("velocity_lag_range") + dt)
        self.filter_restart[reset_buf] = True

    def _spool(self, env: Any):
        spool = env.module_manager.container.get("spool")
        if spool is None:
            raise RuntimeError("spool_state observation requires the 'create_spool' module.")
        return spool

    def get_obs_dim(self, env: Any) -> int:
        return 2 * len(self._spool(env).model_tendon_indices)

    def compute_observation(self, env: Any, out: torch.Tensor) -> None:
        spool = self._spool(env)
        robot = env.module_manager.container.robot
        dofs = spool.dof_indices
        out[:] = _spool_state_obs_jit(robot.get_joint_pos_buf[:, dofs], robot.joint_velocity[:, dofs], spool.pully_radius)
        if self.angle_offset is None:
            return
        n = self.angle_offset.shape[1]
        omega = out[:, n:]
        lagged = self.velocity_filtered + self.velocity_alpha * (omega - self.velocity_filtered)
        self.velocity_filtered = torch.where(self.filter_restart.unsqueeze(1), omega, lagged)
        self.filter_restart[:] = False
        out[:, :n] += self.angle_offset
        out[:, n:] = self.velocity_filtered + self.velocity_offset

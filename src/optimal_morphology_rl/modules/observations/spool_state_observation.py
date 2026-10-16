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

    ``noise.offset_range`` (rad, optional) adds an encoder zero offset to the observed ``theta_m``: one
    value per env and tendon, drawn uniformly on every reset and constant over the episode, so the
    observed 0 is not necessarily the true hand-open spool angle. ``omega_m`` is unaffected and the
    spool itself (its command and dynamics) is unchanged; this is observation noise only.

    Config shape (under ``observation.modules``)::

        spool_state:
          noise:
            offset_range: [-0.1, 0.1]
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        offset_range = (self.config.get("noise") or {}).get("offset_range", [0.0, 0.0])
        if not isinstance(offset_range, (list, tuple)) or len(offset_range) != 2 or float(offset_range[0]) > float(offset_range[1]):
            raise ValueError(f"spool_state 'noise.offset_range' must be a [min, max] list with min <= max, got {offset_range}.")
        self.offset_low, self.offset_high = float(offset_range[0]), float(offset_range[1])
        self.offset: torch.Tensor | None = None  # (num_envs, num_model_tendons), rad

    def reset(self, env: Any) -> None:
        """Draw a new encoder offset for the resetting envs."""
        if self.offset_low == self.offset_high == 0.0:
            return
        reset_buf = env.module_manager.container.get("reset_buf")
        if reset_buf is None:
            return
        if self.offset is None:
            num_tendons = len(self._spool(env).model_tendon_indices)
            self.offset = torch.zeros((reset_buf.shape[0], num_tendons), device=reset_buf.device)
        sample = self.offset_low + (self.offset_high - self.offset_low) * torch.rand_like(self.offset)
        self.offset[reset_buf] = sample[reset_buf]

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
        if self.offset is not None:
            out[:, : self.offset.shape[1]] += self.offset

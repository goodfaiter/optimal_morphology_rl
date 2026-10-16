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
    """

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

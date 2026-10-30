"""Observation source for reward-object pose and velocity in the robot frame."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.create_kinematic_sensor_module import (
    find_kinematic_sensor,
)
from optimal_morphology_rl.modules.observations.observation_base_module import (
    ObservationBaseModule,
)
from optimal_morphology_rl.modules.observations.observation_jit_helpers import (
    _quaternion_to_6d_jit,
)
from optimal_morphology_rl.modules.observations.observation_manager_module import (
    register_observation,
)

# Signal name -> size, in output order.
_SIGNALS = {"pos": 3, "rot": 6, "lin_vel": 3, "ang_vel": 3}


@register_observation("object_state")
class ObjectStateObservation(ObservationBaseModule):
    """Object pose and velocity expressed in the robot base frame.

    ``signals`` (optional, default all) selects and orders the parts: ``pos`` (m), ``rot`` (6D rotation),
    ``lin_vel`` (m/s), ``ang_vel`` (rad/s).

    Config::

        object_state:
          signals: [rot, lin_vel, ang_vel]
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.signals = list(self.config.get("signals", list(_SIGNALS)))
        unknown = [s for s in self.signals if s not in _SIGNALS]
        if unknown or not self.signals:
            raise RuntimeError(f"object_state observation: 'signals' must be a non-empty subset of {list(_SIGNALS)}, got {self.signals}.")

    def get_obs_dim(self, env: Any) -> int:
        return sum(_SIGNALS[s] for s in self.signals)

    def compute_observation(self, env: Any, out: torch.Tensor) -> None:
        container = env.module_manager.container
        sensor = find_kinematic_sensor(container, container.reward_object.name)
        if sensor is None:
            raise RuntimeError(
                "object_state observation: no kinematic sensor for the reward object. "
                "Ensure create_kinematic_sensor is listed."
            )
        parts = {
            "pos": lambda: sensor.pos_in_robot,
            "rot": lambda: _quaternion_to_6d_jit(sensor.quat_sensor_to_robot),
            "lin_vel": lambda: sensor.linear_velocity_in_robot,
            "ang_vel": lambda: sensor.angular_velocity_in_robot,
        }
        out[:] = torch.cat([parts[s]() for s in self.signals], dim=-1)

"""Observation source for reward-object pose and velocity in the robot frame."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.observations.observation_jit_helpers import (
    _object_state_obs_jit,
)
from optimal_morphology_rl.modules.observations.observation_base_module import (
    ObservationBaseModule,
)
from optimal_morphology_rl.modules.observations.observation_manager_module import (
    register_observation,
)


@register_observation("object_state")
class ObjectStateObservation(ObservationBaseModule):
    """Object pose and velocity expressed in the robot base frame."""

    def get_obs_dim(self, env: Any) -> int:
        return 3 + 6 + 3 + 3  # pos, 6d rot, lin vel, ang vel

    def compute_observation(self, env: Any, out: torch.Tensor) -> None:
        container = env.module_manager.container
        sensor = container.kinematic_sensor

        out[:] = _object_state_obs_jit(
            sensor.pos_in_robot,
            sensor.quat_sensor_to_robot,
            sensor.linear_velocity_in_robot,
            sensor.angular_velocity_in_robot,
        )

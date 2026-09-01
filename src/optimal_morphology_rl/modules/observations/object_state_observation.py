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
        robot = container.robot
        sensor = container.kinematic_sensor
        robot_state = robot.get_state()

        out[:] = _object_state_obs_jit(
            robot_state["quat_robot_to_world"],
            robot_state["robot_pos_in_world"],
            sensor.pos_in_world,
            sensor.quat_sensor_to_world,
            sensor.linear_velocity_world,
            sensor.angular_velocity_world,
        )

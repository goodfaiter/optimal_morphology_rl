"""Observation source for reward-object pose and velocity in the robot frame."""

from __future__ import annotations

from typing import Any

import torch
from vlearn.torch_utils.torch_jit_utils import quat_conjugate, quat_mul, quat_rotate_inverse

from optimal_morphology_rl.helpers.numpy_vlearn import quaternion_to_6d
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

        quat_robot_to_world = robot_state["quat_robot_to_world"]
        quat_world_to_robot = quat_conjugate(quat_robot_to_world)
        robot_pos_world = robot_state["robot_pos_in_world"]

        object_pos_world = sensor.pos_in_world
        object_quat_world = sensor.quat_sensor_to_world
        object_lin_vel_world = sensor.linear_velocity_world
        object_ang_vel_world = sensor.angular_velocity_world

        object_position_in_robot_frame = quat_rotate_inverse(
            quat_robot_to_world, object_pos_world - robot_pos_world
        )
        _6d_object_to_robot = quaternion_to_6d(
            quat_mul(quat_world_to_robot, object_quat_world)
        )
        object_linear_velocity_in_robot_frame = quat_rotate_inverse(
            quat_robot_to_world, object_lin_vel_world
        )
        object_angular_velocity_in_robot_frame = quat_rotate_inverse(
            quat_robot_to_world, object_ang_vel_world
        )

        offset = 0
        out[:, offset : offset + 3] = object_position_in_robot_frame
        offset += 3
        out[:, offset : offset + 6] = _6d_object_to_robot
        offset += 6
        out[:, offset : offset + 3] = object_linear_velocity_in_robot_frame
        offset += 3
        out[:, offset : offset + 3] = object_angular_velocity_in_robot_frame

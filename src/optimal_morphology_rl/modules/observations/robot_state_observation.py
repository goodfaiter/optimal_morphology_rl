"""Observation source for robot base / DOF state."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.observations.observation_base_module import (
    ObservationBaseModule,
)
from optimal_morphology_rl.modules.observations.observation_manager_module import (
    register_observation,
)


@register_observation("robot_state")
class RobotStateObservation(ObservationBaseModule):
    """Robot state: optional base velocity, DOF positions/velocities, last action."""

    def get_obs_dim(self, env: Any) -> int:
        robot = env.module_manager.container.robot
        dim = 0
        if not robot.fixed_hand:
            dim += 9  # gravity (3) + lin vel (3) + ang vel (3)
        dim += robot.get_num_dofs()  # dof pos
        dim += robot.get_num_dofs()  # dof vel
        dim += robot.get_num_actions()  # last action
        return dim

    def compute_observation(self, env: Any, out: torch.Tensor) -> None:
        robot = env.module_manager.container.robot
        robot_state = robot.get_state()

        offset = 0
        if not robot.fixed_hand:
            out[:, offset : offset + 3] = robot_state["gravity_vector_in_robot_frame"]
            offset += 3
            out[:, offset : offset + 3] = robot_state[
                "robot_linear_velocity_in_robot_frame"
            ]
            offset += 3
            out[:, offset : offset + 3] = robot_state[
                "robot_angular_velocity_in_robot_frame"
            ]
            offset += 3

        num_dofs = robot.get_num_dofs()
        out[:, offset : offset + num_dofs] = robot_state["dof_pos_buf"]
        offset += num_dofs
        out[:, offset : offset + num_dofs] = robot_state["dof_vel_buf"]
        offset += num_dofs
        out[:, offset : offset + robot.get_num_actions()] = env.act_buf

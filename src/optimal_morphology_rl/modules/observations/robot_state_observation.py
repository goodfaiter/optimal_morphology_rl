"""Observation source for robot base / DOF state."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.observations.observation_jit_helpers import (
    _robot_state_obs_jit,
)
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

        if robot.fixed_hand:
            gravity = torch.empty((env.total_num_envs, 0), device=env.device, dtype=torch.float32)
            lin_vel = gravity
            ang_vel = gravity
        else:
            gravity = robot_state["gravity_vector_in_robot_frame"]
            lin_vel = robot_state["robot_linear_velocity_in_robot_frame"]
            ang_vel = robot_state["robot_angular_velocity_in_robot_frame"]

        out[:] = _robot_state_obs_jit(
            gravity,
            lin_vel,
            ang_vel,
            robot_state["dof_pos_buf"],
            robot_state["dof_vel_buf"],
            env.act_buf,
            robot.fixed_hand,
        )

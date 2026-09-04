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


@torch.jit.script
def _robot_state_obs_extended_jit(
    gravity: torch.Tensor,
    lin_vel: torch.Tensor,
    ang_vel: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Concatenate robot-state observation components."""
    return torch.cat([gravity, lin_vel, ang_vel, dof_pos, dof_vel, actions], dim=-1)


@torch.jit.script
def _robot_state_obs_jit(
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Concatenate robot-state observation components."""
    return torch.cat([dof_pos, dof_vel, actions], dim=-1)


@register_observation("robot_state")
class RobotStateObservation(ObservationBaseModule):
    """Robot state: optional base velocity, DOF positions/velocities, last action."""

    def get_obs_dim(self, env: Any) -> int:
        container = env.module_manager.container
        robot = container.robot
        num_dof_states = robot.num_tendons if robot.use_tendon else robot.num_joints
        dim = 0
        if not robot.fixed_hand:
            dim += 9  # gravity (3) + lin vel (3) + ang vel (3)
        dim += num_dof_states  # dof pos
        dim += num_dof_states  # dof vel
        dim += container.num_actions  # last action
        return dim

    def compute_observation(self, env: Any, out: torch.Tensor) -> None:
        container = env.module_manager.container
        robot = container.robot
        robot_state = container.robot_state

        if not robot.fixed_hand:
            gravity = robot_state["gravity_vector_in_robot_frame"]
            lin_vel = robot_state["robot_linear_velocity_in_robot_frame"]
            ang_vel = robot_state["robot_angular_velocity_in_robot_frame"]
            out[:] = _robot_state_obs_extended_jit(
                gravity, lin_vel, ang_vel, robot_state["dof_pos_buf"], robot_state["dof_vel_buf"], env.act_buf
            )
        else:
            out[:] = _robot_state_obs_jit(robot_state["dof_pos_buf"], robot_state["dof_vel_buf"], env.act_buf)

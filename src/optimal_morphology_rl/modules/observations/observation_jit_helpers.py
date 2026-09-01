"""TorchScript helpers for observation computations."""

from __future__ import annotations

import torch
from vlearn.torch_utils.torch_jit_utils import (
    quat_conjugate,
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
)


@torch.jit.script
def _quaternion_to_6d_jit(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (x, y, z, w) to 6D rotation representation.

    The 6D representation is the first two columns of the rotation matrix.
    """
    num_envs = q.shape[0]
    device = q.device

    e1 = torch.tensor([1.0, 0.0, 0.0], device=device).unsqueeze(0).expand(num_envs, 3)
    e2 = torch.tensor([0.0, 1.0, 0.0], device=device).unsqueeze(0).expand(num_envs, 3)

    col1 = quat_rotate(q, e1)
    col2 = quat_rotate(q, e2)
    return torch.cat([col1, col2], dim=-1)


@torch.jit.script
def _robot_state_obs_jit(
    gravity: torch.Tensor,
    lin_vel: torch.Tensor,
    ang_vel: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    actions: torch.Tensor,
    fixed_hand: bool,
) -> torch.Tensor:
    """Concatenate robot-state observation components."""
    if fixed_hand:
        return torch.cat([dof_pos, dof_vel, actions], dim=-1)
    return torch.cat([gravity, lin_vel, ang_vel, dof_pos, dof_vel, actions], dim=-1)


@torch.jit.script
def _object_state_obs_jit(
    quat_robot_to_world: torch.Tensor,
    robot_pos_in_world: torch.Tensor,
    object_pos_in_world: torch.Tensor,
    object_quat_in_world: torch.Tensor,
    object_lin_vel_in_world: torch.Tensor,
    object_ang_vel_in_world: torch.Tensor,
) -> torch.Tensor:
    """Compute object-state observation in the robot frame."""
    quat_world_to_robot = quat_conjugate(quat_robot_to_world)

    object_pos_in_robot = quat_rotate_inverse(quat_robot_to_world, object_pos_in_world - robot_pos_in_world)
    object_quat_in_robot = quat_mul(quat_world_to_robot, object_quat_in_world)
    _6d_object_to_robot = _quaternion_to_6d_jit(object_quat_in_robot)

    object_lin_vel_in_robot = quat_rotate_inverse(quat_robot_to_world, object_lin_vel_in_world)
    object_ang_vel_in_robot = quat_rotate_inverse(quat_robot_to_world, object_ang_vel_in_world)

    return torch.cat(
        [
            object_pos_in_robot,
            _6d_object_to_robot,
            object_lin_vel_in_robot,
            object_ang_vel_in_robot,
        ],
        dim=-1,
    )


@torch.jit.script
def _goal_state_obs_jit(
    quat_robot_to_world: torch.Tensor,
    robot_pos_in_world: torch.Tensor,
    goal_pos_in_world: torch.Tensor,
    goal_quat_object_to_world: torch.Tensor,
) -> torch.Tensor:
    """Compute goal-state observation in the robot frame."""
    quat_world_to_robot = quat_conjugate(quat_robot_to_world)

    goal_pos_in_robot = quat_rotate_inverse(quat_robot_to_world, goal_pos_in_world - robot_pos_in_world)
    goal_quat_in_robot = quat_mul(quat_world_to_robot, goal_quat_object_to_world)
    _6d_goal_to_robot = _quaternion_to_6d_jit(goal_quat_in_robot)

    return torch.cat([goal_pos_in_robot, _6d_goal_to_robot], dim=-1)

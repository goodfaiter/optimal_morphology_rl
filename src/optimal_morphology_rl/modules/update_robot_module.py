"""Module that refreshes robot state buffers after the physics step."""

from __future__ import annotations

from functools import partial
from typing import Any

import torch
from vlearn.torch_utils.torch_jit_utils import quat_rotate_inverse
import vlearn as v

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.create_robot_module import Robot
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module
from optimal_morphology_rl.modules.observations.observation_jit_helpers import (
    _quaternion_to_6d_jit,
)


def _allocate_read_buffers(robot: Robot, total_num_envs: int, device: torch.device) -> None:
    """Allocate buffers used to read robot state from simulation."""
    robot.get_joint_pos_buf = torch.zeros((total_num_envs, robot.num_joints), device=device, dtype=torch.float32)
    robot.get_joint_vel_buf = torch.zeros((total_num_envs, robot.num_joints), device=device, dtype=torch.float32)
    robot.get_root_transform_buf = torch.zeros((total_num_envs, 7), device=device, dtype=torch.float32)
    robot.get_root_vel_buf = torch.zeros((total_num_envs, 6), device=device, dtype=torch.float32)

    robot.robot_pos_in_world = torch.zeros((total_num_envs, 3), device=device, dtype=torch.float32)
    robot.quat_robot_to_world = torch.zeros((total_num_envs, 4), device=device, dtype=torch.float32)
    robot._6d_robot_to_world = torch.zeros((total_num_envs, 6), device=device, dtype=torch.float32)
    robot.robot_linear_velocity_in_world = torch.zeros((total_num_envs, 3), device=device, dtype=torch.float32)
    robot.robot_angular_velocity_in_world = torch.zeros((total_num_envs, 3), device=device, dtype=torch.float32)

    robot.gravity_direction_world = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=torch.float32).repeat(total_num_envs, 1)
    robot.gravity_vector_in_robot_frame = torch.zeros((total_num_envs, 3), device=device, dtype=torch.float32)
    robot.robot_linear_velocity_in_robot_frame = torch.zeros((total_num_envs, 3), device=device, dtype=torch.float32)
    robot.robot_angular_velocity_in_robot_frame = torch.zeros((total_num_envs, 3), device=device, dtype=torch.float32)

    if robot.use_tendon:
        robot.get_tendon_lengths_buf = torch.zeros((total_num_envs, robot.num_tendons), dtype=torch.float32, device=device)
        robot.get_tendon_vel_buf = torch.zeros((total_num_envs, robot.num_tendons), dtype=torch.float32, device=device)

    if robot.num_distal_links > 0:
        robot.distal_link_transform_buf = torch.zeros(
            (robot.num_distal_links, total_num_envs, 7),
            device=device,
            dtype=torch.float32,
        )
        robot.distal_link_pos_buf = torch.zeros(
            (robot.num_distal_links, total_num_envs, 3),
            device=device,
            dtype=torch.float32,
        )


def _create_read_gpu_commands(robot: Robot, env_group: Any, gym: v.Gym) -> None:
    """Create GPU commands for reading robot state."""
    get_kin_cmd = env_group.create_articulation_kinematic_state_command(
        v.wrap_gpu_buffer(robot.get_joint_pos_buf),
        v.wrap_gpu_buffer(robot.get_joint_vel_buf),
        v.wrap_gpu_buffer(robot.get_root_transform_buf),
        v.wrap_gpu_buffer(robot.get_root_vel_buf),
        robot.arti_handle,
        (0, robot.num_joints),
        (0, 1),
    )
    robot.gpu_get_kinematic_state_command_array = gym.create_gpu_array([get_kin_cmd])

    if robot.num_distal_links > 0:
        distal_transform_cmds = []
        for i, link_index in enumerate(robot.distal_link_indices):
            distal_transform_cmds.append(
                env_group.create_link_transform_command(
                    v.wrap_gpu_buffer(robot.distal_link_transform_buf[i, :, :]),
                    robot.arti_handle,
                    (link_index, link_index + 1),
                )
            )
        robot.gpu_get_distal_link_transforms_cmd_arr = gym.create_gpu_array(distal_transform_cmds)

    if robot.use_tendon:
        get_tendon_lengths_cmd = env_group.create_spatial_tendon_state_command(
            v.SpatialTendonState.LENGTH,
            v.wrap_gpu_buffer(robot.get_tendon_lengths_buf),
            robot.arti_handle,
            (0, robot.num_tendons),
        )
        robot.gpu_get_tendon_lengths_command_array = gym.create_gpu_array([get_tendon_lengths_cmd])

        get_tendon_vel_cmd = env_group.create_spatial_tendon_state_command(
            v.SpatialTendonState.VELOCITY,
            v.wrap_gpu_buffer(robot.get_tendon_vel_buf),
            robot.arti_handle,
            (0, robot.num_tendons),
        )
        robot.gpu_get_tendon_velocities_command_array = gym.create_gpu_array([get_tendon_vel_cmd])


def _refresh_buffers(robot: Robot, gym: v.Gym) -> None:
    """Refresh robot kinematic state from simulation."""
    gym.get_articulation_kinematic_states(robot.gpu_get_kinematic_state_command_array)
    if robot.num_distal_links > 0:
        gym.get_link_transforms(robot.gpu_get_distal_link_transforms_cmd_arr)
        robot.distal_link_pos_buf[:] = robot.distal_link_transform_buf[:, :, 4:7]
    if robot.use_tendon:
        gym.get_spatial_tendon_states(robot.gpu_get_tendon_lengths_command_array)
        gym.get_spatial_tendon_states(robot.gpu_get_tendon_velocities_command_array)


def _get_state(robot: Robot) -> dict[str, torch.Tensor]:
    """Update and return robot-derived observation tensors."""
    robot.robot_pos_in_world[:] = robot.get_root_transform_buf[:, 4:7]
    robot.quat_robot_to_world[:] = robot.get_root_transform_buf[:, 0:4]
    robot._6d_robot_to_world[:] = _quaternion_to_6d_jit(robot.quat_robot_to_world)
    robot.robot_linear_velocity_in_world[:] = robot.get_root_vel_buf[:, 3:6]
    robot.robot_angular_velocity_in_world[:] = robot.get_root_vel_buf[:, :3]

    robot.gravity_vector_in_robot_frame[:] = quat_rotate_inverse(robot.quat_robot_to_world, robot.gravity_direction_world)
    robot.robot_linear_velocity_in_robot_frame[:] = quat_rotate_inverse(robot.quat_robot_to_world, robot.robot_linear_velocity_in_world)
    robot.robot_angular_velocity_in_robot_frame[:] = quat_rotate_inverse(robot.quat_robot_to_world, robot.robot_angular_velocity_in_world)

    state = {
        "robot_pos_in_world": robot.robot_pos_in_world,
        "quat_robot_to_world": robot.quat_robot_to_world,
        "_6d_robot_to_world": robot._6d_robot_to_world,
        "robot_linear_velocity_in_world": robot.robot_linear_velocity_in_world,
        "robot_angular_velocity_in_world": robot.robot_angular_velocity_in_world,
        "gravity_vector_in_robot_frame": robot.gravity_vector_in_robot_frame,
        "robot_linear_velocity_in_robot_frame": robot.robot_linear_velocity_in_robot_frame,
        "robot_angular_velocity_in_robot_frame": robot.robot_angular_velocity_in_robot_frame,
        "get_root_transform_buf": robot.get_root_transform_buf,
        "get_root_vel_buf": robot.get_root_vel_buf,
        "set_motor_cmd_buf": robot.set_motor_cmd_buf,
    }
    if robot.use_tendon:
        state["dof_pos_buf"] = robot.get_tendon_lengths_buf
        state["dof_vel_buf"] = robot.get_tendon_vel_buf
    else:
        state["dof_pos_buf"] = robot.get_joint_pos_buf
        state["dof_vel_buf"] = robot.get_joint_vel_buf
    return state


@register_module("update_robot")
class UpdateRobotModule(BaseModule):
    """Refreshes robot state buffers for the robot created by ``create_robot``."""

    def post_finalize(self, container: ModuleContainer) -> None:
        """Allocate read buffers and create read GPU commands."""
        robot = container.robot
        _allocate_read_buffers(robot, container.total_num_envs, container.device)
        _create_read_gpu_commands(robot, container.env_group, container.gym)

        # Bind state methods onto the robot so legacy callers keep working.
        robot.refresh_buffers = partial(_refresh_buffers, robot)
        robot.get_state = partial(_get_state, robot)

    def step(self, container: ModuleContainer) -> None:
        """Refresh robot state buffers."""
        robot = container.get("robot")
        if robot is None:
            return
        robot.refresh_buffers(container.gym)

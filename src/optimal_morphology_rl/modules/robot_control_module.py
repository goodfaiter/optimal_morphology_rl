"""Module that handles action scaling, robot control buffers, and control commands."""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import torch
from vlearn.spaces import Box
from vlearn.torch_utils.torch_jit_utils import quat_rotate
import vlearn as v

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.create_robot_module import Robot
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module
from optimal_morphology_rl.helpers.numpy_vlearn import random_uniform_quaternion


# ---------------------------------------------------------------------------
# Buffer allocation
# ---------------------------------------------------------------------------
def _allocate_control_buffers(robot: Robot, total_num_envs: int, device: torch.device) -> None:
    """Allocate buffers used for control and reset."""
    robot.reset_joint_pos_buf = torch.zeros(
        (total_num_envs, robot.num_joints), device=device, dtype=torch.float32
    )
    robot.reset_joint_vel_buf = torch.zeros(
        (total_num_envs, robot.num_joints), device=device, dtype=torch.float32
    )
    robot.reset_root_transform_buf = torch.zeros(
        (total_num_envs, 7), device=device, dtype=torch.float32
    )
    robot.reset_root_vel_buf = torch.zeros(
        (total_num_envs, 6), device=device, dtype=torch.float32
    )

    robot.set_joint_pos_buf = torch.zeros(
        (total_num_envs, 0), device=device, dtype=torch.float32
    )
    robot.set_joint_vel_buf = torch.zeros(
        (total_num_envs, 0), device=device, dtype=torch.float32
    )
    robot.set_root_transform_buf = torch.zeros(
        (total_num_envs, 7), device=device, dtype=torch.float32
    )
    robot.set_root_vel_buf = torch.zeros(
        (total_num_envs, 6), device=device, dtype=torch.float32
    )

    robot.set_motor_cmd_buf = torch.zeros(
        (total_num_envs, robot.num_motors), device=device, dtype=torch.float32
    )
    robot.set_force_torque_buf = torch.zeros(
        (total_num_envs, robot.num_links, 6), dtype=torch.float32, device=device
    )

    # Rigid material property buffers are scalar per material.
    robot.set_static_friction_buf = torch.zeros(1, dtype=torch.float32, device=device)
    robot.set_dynamic_friction_buf = torch.zeros(1, dtype=torch.float32, device=device)

    if robot.use_tendon:
        robot.set_tendon_controls_buf = torch.zeros(
            (total_num_envs, robot.num_tendons), dtype=torch.float32, device=device
        )


# ---------------------------------------------------------------------------
# GPU commands
# ---------------------------------------------------------------------------
def _create_control_gpu_commands(
    robot: Robot,
    env_group: Any,
    gym: v.Gym,
    reset_buf: torch.Tensor,
    inverse_reset_buf: torch.Tensor,
) -> None:
    """Create GPU commands for control and reset."""
    reset_kin_cmd = env_group.create_articulation_kinematic_state_command(
        v.wrap_gpu_buffer(robot.reset_joint_pos_buf),
        v.wrap_gpu_buffer(robot.reset_joint_vel_buf),
        v.wrap_gpu_buffer(robot.reset_root_transform_buf),
        v.wrap_gpu_buffer(robot.reset_root_vel_buf),
        robot.arti_handle,
        (0, robot.num_joints),
        (0, 1),
        masks_buffer=v.wrap_gpu_buffer(reset_buf),
    )
    robot.gpu_reset_kinematic_state_command_array = gym.create_gpu_array([reset_kin_cmd])

    set_kin_cmd = env_group.create_articulation_kinematic_state_command(
        v.wrap_gpu_buffer(robot.set_joint_pos_buf),
        v.wrap_gpu_buffer(robot.set_joint_vel_buf),
        v.wrap_gpu_buffer(robot.set_root_transform_buf),
        v.wrap_gpu_buffer(robot.set_root_vel_buf),
        robot.arti_handle,
        (0, 0),
        (0, 1),
        masks_buffer=v.wrap_gpu_buffer(inverse_reset_buf),
    )
    robot.gpu_set_kinematic_state_command_array = gym.create_gpu_array([set_kin_cmd])

    set_motor_cmd = env_group.create_motor_control_command(
        v.wrap_gpu_buffer(robot.set_motor_cmd_buf),
        robot.arti_handle,
        index_range=[0, robot.num_motors],
    )
    robot.gpu_set_motor_control_command_array = gym.create_gpu_array([set_motor_cmd])

    if robot.use_tendon:
        set_tendon_cmd = env_group.create_spatial_tendon_control_command(
            v.wrap_gpu_buffer(robot.set_tendon_controls_buf), robot.arti_handle
        )
        robot.gpu_set_tendon_control_command_array = gym.create_gpu_array([set_tendon_cmd])

    # Gravity compensation external force command.
    set_force_torque_cmd = env_group.create_link_external_force_command(
        v.wrap_gpu_buffer(robot.set_force_torque_buf),
        robot.arti_handle,
        [0, robot.num_links],
        force_type=v.ForceType.FORCE_TORQUE,
    )
    robot.set_force_torque_cmd_arr = gym.create_gpu_array([set_force_torque_cmd])

    # Rigid material commands.
    set_static_friction_cmd = env_group.create_rigid_material_property_command(
        v.RigidMaterialProperty.STATIC_FRICTION,
        v.wrap_gpu_buffer(robot.set_static_friction_buf),
        robot.rigid_mat_handle,
        v.wrap_gpu_buffer(reset_buf),
    )
    set_dynamic_friction_cmd = env_group.create_rigid_material_property_command(
        v.RigidMaterialProperty.DYNAMIC_FRICTION,
        v.wrap_gpu_buffer(robot.set_dynamic_friction_buf),
        robot.rigid_mat_handle,
        v.wrap_gpu_buffer(reset_buf),
    )
    robot.gpu_set_friction_cmd = gym.create_gpu_array(
        [set_static_friction_cmd, set_dynamic_friction_cmd]
    )


@register_module("robot_control")
class RobotControlModule(BaseModule):
    """Owns robot control buffer allocation and per-step control commands.

    Expects ``container.robot`` to be populated by the ``create_robot`` module
    and ``container.scaled_act_buf`` to be populated by ``process_actions``.
    """

    def finalize(self, container: ModuleContainer) -> None:
        """Set the environment action space from the robot DOFs."""
        if container.get("robot") is None:
            raise RuntimeError(
                "RobotControlModule requires 'robot' in the shared container. "
                "Ensure the 'create_robot' module is listed before 'robot_control'."
            )

        env = container.env
        robot = container.robot
        num_actions = robot.get_num_actions()

        env.action_space = Box(
            low=np.full(num_actions, -1.0, dtype=np.float32),
            high=np.full(num_actions, 1.0, dtype=np.float32),
            dtype=np.float32,
        )

    def post_finalize(self, container: ModuleContainer) -> None:
        """Allocate control buffers and create GPU commands."""
        env = container.env
        robot = container.robot
        total_num_envs = container.total_num_envs
        device = container.device

        if container.get("act_buf") is None or container.get("scaled_act_buf") is None:
            raise RuntimeError(
                "RobotControlModule requires 'act_buf' and 'scaled_act_buf' in the "
                "shared container. Ensure 'process_actions' is listed before "
                "'robot_control' in pre_physics_step_modules."
            )
        if container.get("inverse_reset_buf") is None:
            raise RuntimeError(
                "RobotControlModule requires 'inverse_reset_buf' in the shared "
                "container. Ensure 'termination' is loaded."
            )

        _allocate_control_buffers(robot, total_num_envs, device)

        _create_control_gpu_commands(
            robot,
            container.env_group,
            container.gym,
            container.reset_buf,
            container.inverse_reset_buf,
        )

    def step(self, container: ModuleContainer) -> None:
        """Apply wrist velocity, joint motor commands, and gravity compensation."""
        robot = container.robot
        robot.pre_physics_step(container.gym)
        gym = container.gym

        if not robot.fixed_hand:
            robot.set_root_transform_buf[:] = robot.get_root_transform_buf
            local_root_vel = torch.clamp(
                robot.scaled_act_buf[:, robot.root_slice],
                -robot.max_velocity,
                robot.max_velocity,
            )
            quat_robot_to_world = robot.get_root_transform_buf[:, 0:4]
            robot.set_root_vel_buf[:, :3] = quat_rotate(
                quat_robot_to_world, local_root_vel[:, :3]
            )
            robot.set_root_vel_buf[:, 3:] = quat_rotate(
                quat_robot_to_world, local_root_vel[:, 3:]
            )
            gym.set_articulation_kinematic_states(robot.gpu_set_kinematic_state_command_array)

        robot.set_motor_cmd_buf[:] = 0.0

        if robot.use_tendon:
            robot.set_tendon_controls_buf[:] = torch.clamp(
                robot.scaled_act_buf[:, robot.dof_slice], 0.0, None
            )
            gym.set_spatial_tendon_forces(robot.gpu_set_tendon_control_command_array)
        else:
            robot.set_motor_cmd_buf[:] = torch.clamp(
                robot.scaled_act_buf[:, robot.dof_slice], 0.0, None
            )

        # Antagonistic spring on all joints.
        robot.set_motor_cmd_buf[:] += -0.1 * robot.get_joint_pos_buf
        gym.set_motor_forces(robot.gpu_set_motor_control_command_array)

        # Gravity compensation on base link.
        robot.set_force_torque_buf[:, :, 2] = 9.81 * robot.link_masses
        gym.set_link_external_forces(robot.set_force_torque_cmd_arr)

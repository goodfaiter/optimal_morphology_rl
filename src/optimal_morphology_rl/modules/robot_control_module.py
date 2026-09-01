"""Module that handles action scaling, robot control buffers, and control commands."""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import torch
from vlearn.spaces import Box
from vlearn.torch_utils.torch_jit_utils import scale, quat_rotate
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


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------
def _reset_idx(
    robot: Robot,
    gym: v.Gym,
    reset_buf: torch.Tensor,
    device: torch.device,
    fric_coeff: float | None = None,
    randomize_pose: bool = False,
) -> None:
    """Reset robot kinematic state for the given reset indices."""
    robot.reset_joint_pos_buf[reset_buf, :] = 0.0
    robot.reset_joint_vel_buf[reset_buf, :] = 0.0
    if robot.fixed_hand:
        robot.reset_root_transform_buf[reset_buf, 4:] = torch.tensor(
            [[-0.1, -0.15, 0.1]], device=device
        )
        robot.reset_root_transform_buf[reset_buf, :4] = torch.tensor(
            [0.6963642, 0.1227878, -0.1227878, 0.6963642], device=device
        )
    else:
        if randomize_pose:
            n_reset = reset_buf.sum().item()
            robot.reset_root_transform_buf[reset_buf, :4] = random_uniform_quaternion(
                n_reset, device=device, dtype=torch.float32
            )
            robot.reset_root_transform_buf[reset_buf, 4] = -0.1
            robot.reset_root_transform_buf[reset_buf, 5] = (
                torch.rand(n_reset, device=device) * 0.3 - 0.15
            )
            robot.reset_root_transform_buf[reset_buf, 6] = (
                torch.rand(n_reset, device=device) * 0.2 + 0.1
            )
        else:
            robot.reset_root_transform_buf[reset_buf, 4:] = torch.tensor(
                [[-0.1, -0.15, 0.2]], device=device
            )
            robot.reset_root_transform_buf[reset_buf, :4] = torch.tensor(
                [0.0, 0.0, 0.0, 1.0], device=device
            )
    robot.reset_root_vel_buf[reset_buf, :] = 0.0
    gym.set_articulation_kinematic_states(robot.gpu_reset_kinematic_state_command_array)

    total_num_envs = reset_buf.shape[0]
    if total_num_envs != 1 and fric_coeff is None:
        static_friction = torch.rand(1, device=device).item() * 0.9 + 0.1
    else:
        static_friction = 0.1 if fric_coeff is None else fric_coeff
    dynamic_friction = static_friction * 0.75

    robot.set_static_friction_buf[0] = static_friction * 2.0
    robot.set_dynamic_friction_buf[0] = dynamic_friction * 2.0
    gym.set_rigid_material_properties(robot.gpu_set_friction_cmd)


# ---------------------------------------------------------------------------
# Pre-physics step
# ---------------------------------------------------------------------------
def _pre_physics_step(
    robot: Robot,
    gym: v.Gym,
    act_buf: torch.Tensor,
) -> None:
    """Apply wrist velocity, joint motor commands, and gravity compensation."""
    robot.scaled_act_buf[:, robot.root_slice] = scale(
        act_buf[:, robot.root_slice],
        -robot.velocity_scale[robot.root_slice],
        robot.velocity_scale[robot.root_slice],
    )
    robot.scaled_act_buf[:, robot.dof_slice] = scale(
        act_buf[:, robot.dof_slice],
        robot.min_revolute_scale,
        robot.max_revolute_scale,
    )

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


@register_module("robot_control")
class RobotControlModule(BaseModule):
    """Owns robot action scaling, control buffer allocation, and per-step control.

    Expects ``container.robot`` to be populated by the ``create_robot`` module.
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
        """Allocate action/control buffers and create GPU commands."""
        env = container.env
        robot = container.robot
        total_num_envs = container.total_num_envs
        device = container.device

        container.act_buf = torch.zeros(
            (total_num_envs,) + env.action_space.shape,
            device=device,
            dtype=torch.float32,
        )

        _allocate_control_buffers(robot, total_num_envs, device)

        env.inverse_reset_buf = torch.zeros(
            total_num_envs, device=device, dtype=torch.bool
        )
        env.last_act_buf = torch.zeros_like(env.act_buf)
        env.scaled_act_buf = torch.zeros_like(env.act_buf)
        robot.scaled_act_buf = env.scaled_act_buf

        _create_control_gpu_commands(
            robot,
            container.env_group,
            container.gym,
            container.reset_buf,
            env.inverse_reset_buf,
        )

        # Bind control methods onto the robot so legacy callers keep working.
        robot.reset_idx = partial(_reset_idx, robot)
        robot.pre_physics_step = partial(_pre_physics_step, robot)

    def step(self, container: ModuleContainer) -> None:
        """Apply wrist velocity, joint motor commands, and gravity compensation."""
        env = container.env
        robot = container.robot
        env.last_act_buf[:] = env.act_buf[:]
        robot.pre_physics_step(container.gym, env.act_buf)

    def reset(self, container: ModuleContainer) -> None:
        """Reset robot state for the environments selected by the reset buffer."""
        env = container.env
        robot = container.robot
        reset_config = container.get("robot_reset_config", {})

        env.act_buf[container.reset_buf, :] = 0.0
        env.last_act_buf[container.reset_buf, :] = 0.0

        robot.reset_idx(
            container.gym,
            container.reset_buf,
            container.device,
            fric_coeff=reset_config.get("fric_coeff", None),
            randomize_pose=reset_config.get("randomize_pose", False),
        )

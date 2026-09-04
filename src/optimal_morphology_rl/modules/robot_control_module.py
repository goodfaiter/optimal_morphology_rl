"""Module that handles action scaling, robot control buffers, and control commands."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from vlearn.spaces import Box
from vlearn.torch_utils.torch_jit_utils import quat_rotate
import vlearn as v

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


# ---------------------------------------------------------------------------
# Active motor mask
# ---------------------------------------------------------------------------
def build_active_motor_mask(container: ModuleContainer, config: dict[str, Any]) -> None:
    """Build the mask and slices used to map policy actions to robot actuators.

    For tendon-driven hands the action vector maps directly to the tendons.
    For motor-driven hands, motors whose names contain any of the substrings in
    ``passive_motor_substrings`` are passive (spring only); everything else is
    active. Alternatively ``active_motor_substrings`` can be provided to
    explicitly select active motors.

    Results are stored directly on ``container``.
    """
    robot = container.robot
    if robot.art_def is None or robot.num_motors is None:
        raise RuntimeError("create_envs must be called before build_active_motor_mask")

    device = robot.motor_to_joint_dof_index.device
    container.root_slice = slice(0, 6) if not robot.fixed_hand else slice(0, 0)

    if robot.use_tendon:
        # Tendon-driven hands: actions map one-to-one to spatial tendons.
        num_active = robot.num_tendons
        container.num_active_motors = num_active
        container.active_motor_mask = torch.ones(num_active, dtype=torch.bool, device=device)
        container.active_motor_indices = torch.arange(num_active, device=device)
        min_scale = -0.25 * robot.tendon_max_force
        max_scale = 1.0 * robot.tendon_max_force
        container.min_active_motor_scale = torch.full((num_active,), min_scale, device=device)
        container.max_active_motor_scale = torch.full((num_active,), max_scale, device=device)
        if robot.fixed_hand:
            container.active_motor_slice = slice(0, num_active)
        else:
            container.active_motor_slice = slice(6, 6 + num_active)
        return

    active_substrings = config.get("active_motor_substrings")
    passive_substrings = config.get("passive_motor_substrings", ["abd"])

    mask = torch.ones(robot.num_motors, dtype=torch.bool, device=device)
    for i in range(robot.num_motors):
        name = robot.art_def.get_motor_def(i).name.lower()
        if active_substrings is not None:
            mask[i] = any(sub.lower() in name for sub in active_substrings)
        else:
            mask[i] = not any(sub.lower() in name for sub in passive_substrings)

    container.active_motor_mask = mask
    container.active_motor_indices = torch.nonzero(mask, as_tuple=False).flatten()
    container.num_active_motors = int(mask.sum().item())

    min_scale = -1.0 * robot.max_torque
    max_scale = 1.0 * robot.max_torque

    container.min_active_motor_scale = torch.full((container.num_active_motors,), min_scale, device=device)
    container.max_active_motor_scale = torch.full((container.num_active_motors,), max_scale, device=device)

    if robot.fixed_hand:
        container.active_motor_slice = slice(0, container.num_active_motors)
    else:
        container.active_motor_slice = slice(6, 6 + container.num_active_motors)


def get_num_actions(container: ModuleContainer) -> int:
    """Return the number of actions for the robot."""
    robot = container.robot
    num_active = robot.num_tendons if robot.use_tendon else container.num_active_motors
    return num_active if robot.fixed_hand else 6 + num_active


# ---------------------------------------------------------------------------
# Buffer allocation
# ---------------------------------------------------------------------------
def _allocate_control_buffers(container: ModuleContainer) -> None:
    """Allocate buffers used for control and reset."""
    robot = container.robot
    total_num_envs = container.total_num_envs
    device = container.device

    container.reset_joint_pos_buf = torch.zeros((total_num_envs, robot.num_joints), device=device, dtype=torch.float32)
    container.reset_joint_vel_buf = torch.zeros((total_num_envs, robot.num_joints), device=device, dtype=torch.float32)
    container.reset_root_transform_buf = torch.zeros((total_num_envs, 7), device=device, dtype=torch.float32)
    container.reset_root_vel_buf = torch.zeros((total_num_envs, 6), device=device, dtype=torch.float32)

    container.set_joint_pos_buf = torch.zeros((total_num_envs, 0), device=device, dtype=torch.float32)
    container.set_joint_vel_buf = torch.zeros((total_num_envs, 0), device=device, dtype=torch.float32)
    container.set_root_transform_buf = torch.zeros((total_num_envs, 7), device=device, dtype=torch.float32)
    container.set_root_vel_buf = torch.zeros((total_num_envs, 6), device=device, dtype=torch.float32)

    container.set_motor_cmd_buf = torch.zeros((total_num_envs, robot.num_motors), device=device, dtype=torch.float32)
    container.set_force_torque_buf = torch.zeros((total_num_envs, robot.num_links, 6), dtype=torch.float32, device=device)

    # Rigid material property buffers are scalar per material.
    container.set_static_friction_buf = torch.zeros(1, dtype=torch.float32, device=device)
    container.set_dynamic_friction_buf = torch.zeros(1, dtype=torch.float32, device=device)

    if robot.use_tendon:
        container.set_tendon_controls_buf = torch.zeros((total_num_envs, robot.num_tendons), dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# GPU commands
# ---------------------------------------------------------------------------
def _create_control_gpu_commands(container: ModuleContainer) -> None:
    """Create GPU commands for control and reset."""
    robot = container.robot
    env_group = container.env_group
    gym = container.gym
    reset_buf = container.reset_buf
    inverse_reset_buf = container.inverse_reset_buf

    reset_kin_cmd = env_group.create_articulation_kinematic_state_command(
        v.wrap_gpu_buffer(container.reset_joint_pos_buf),
        v.wrap_gpu_buffer(container.reset_joint_vel_buf),
        v.wrap_gpu_buffer(container.reset_root_transform_buf),
        v.wrap_gpu_buffer(container.reset_root_vel_buf),
        robot.arti_handle,
        (0, robot.num_joints),
        (0, 1),
        masks_buffer=v.wrap_gpu_buffer(reset_buf),
    )
    container.gpu_reset_kinematic_state_command_array = gym.create_gpu_array([reset_kin_cmd])

    set_kin_cmd = env_group.create_articulation_kinematic_state_command(
        v.wrap_gpu_buffer(container.set_joint_pos_buf),
        v.wrap_gpu_buffer(container.set_joint_vel_buf),
        v.wrap_gpu_buffer(container.set_root_transform_buf),
        v.wrap_gpu_buffer(container.set_root_vel_buf),
        robot.arti_handle,
        (0, 0),
        (0, 1),
        masks_buffer=v.wrap_gpu_buffer(inverse_reset_buf),
    )
    container.gpu_set_kinematic_state_command_array = gym.create_gpu_array([set_kin_cmd])

    set_motor_cmd = env_group.create_motor_control_command(
        v.wrap_gpu_buffer(container.set_motor_cmd_buf),
        robot.arti_handle,
        index_range=[0, robot.num_motors],
    )
    container.gpu_set_motor_control_command_array = gym.create_gpu_array([set_motor_cmd])

    if robot.use_tendon:
        set_tendon_cmd = env_group.create_spatial_tendon_control_command(
            v.wrap_gpu_buffer(container.set_tendon_controls_buf), robot.arti_handle
        )
        container.gpu_set_tendon_control_command_array = gym.create_gpu_array([set_tendon_cmd])

    # Gravity compensation external force command.
    set_force_torque_cmd = env_group.create_link_external_force_command(
        v.wrap_gpu_buffer(container.set_force_torque_buf),
        robot.arti_handle,
        [0, robot.num_links],
        force_type=v.ForceType.FORCE_TORQUE,
    )
    container.set_force_torque_cmd_arr = gym.create_gpu_array([set_force_torque_cmd])

    # Rigid material commands.
    set_static_friction_cmd = env_group.create_rigid_material_property_command(
        v.RigidMaterialProperty.STATIC_FRICTION,
        v.wrap_gpu_buffer(container.set_static_friction_buf),
        robot.rigid_mat_handle,
        v.wrap_gpu_buffer(reset_buf),
    )
    set_dynamic_friction_cmd = env_group.create_rigid_material_property_command(
        v.RigidMaterialProperty.DYNAMIC_FRICTION,
        v.wrap_gpu_buffer(container.set_dynamic_friction_buf),
        robot.rigid_mat_handle,
        v.wrap_gpu_buffer(reset_buf),
    )
    container.gpu_set_friction_cmd = gym.create_gpu_array([set_static_friction_cmd, set_dynamic_friction_cmd])


@register_module("robot_control")
class RobotControlModule(BaseModule):
    """Owns robot control buffer allocation and per-step control commands.

    Expects ``container.robot`` to be populated by the ``create_robot`` module
    and ``container.scaled_act_buf`` to be populated by ``process_actions``.
    """

    def finalize(self, container: ModuleContainer) -> None:
        """Build the active motor mask and set the environment action space."""
        if container.get("robot") is None:
            raise RuntimeError(
                "RobotControlModule requires 'robot' in the shared container. "
                "Ensure the 'create_robot' module is listed before 'robot_control'."
            )
        if container.get("create_robot_config") is None:
            raise RuntimeError(
                "RobotControlModule requires 'create_robot_config' in the shared container. "
                "Ensure the 'create_robot' module is listed before 'robot_control'."
            )

        build_active_motor_mask(container, container.create_robot_config)

        env = container.env
        num_actions = get_num_actions(container)

        env.action_space = Box(
            low=np.full(num_actions, -1.0, dtype=np.float32),
            high=np.full(num_actions, 1.0, dtype=np.float32),
            dtype=np.float32,
        )
        container.num_actions = num_actions

    def post_finalize(self, container: ModuleContainer) -> None:
        """Allocate control buffers and create GPU commands."""
        if container.get("act_buf") is None or container.get("scaled_act_buf") is None:
            raise RuntimeError(
                "RobotControlModule requires 'act_buf' and 'scaled_act_buf' in the "
                "shared container. Ensure 'process_actions' is listed before "
                "'robot_control' in pre_physics_step_modules."
            )
        if container.get("inverse_reset_buf") is None:
            raise RuntimeError("RobotControlModule requires 'inverse_reset_buf' in the shared container. Ensure 'termination' is loaded.")

        _allocate_control_buffers(container)
        _create_control_gpu_commands(container)

    def step(self, container: ModuleContainer) -> None:
        """Apply wrist velocity, joint motor commands, and gravity compensation."""
        robot = container.robot
        gym = container.gym

        if not robot.fixed_hand:
            container.set_root_transform_buf[:] = robot.get_root_transform_buf
            local_root_vel = torch.clamp(
                container.scaled_act_buf[:, container.root_slice],
                -robot.max_velocity,
                robot.max_velocity,
            )
            quat_robot_to_world = robot.get_root_transform_buf[:, 0:4]
            container.set_root_vel_buf[:, :3] = quat_rotate(quat_robot_to_world, local_root_vel[:, :3])
            container.set_root_vel_buf[:, 3:] = quat_rotate(quat_robot_to_world, local_root_vel[:, 3:])
            gym.set_articulation_kinematic_states(container.gpu_set_kinematic_state_command_array)

        container.set_motor_cmd_buf[:] = 0.0

        if robot.use_tendon:
            container.set_tendon_controls_buf[:] = torch.clamp(container.scaled_act_buf[:, container.active_motor_slice], 0.0, None)
            gym.set_spatial_tendon_forces(container.gpu_set_tendon_control_command_array)
        else:
            # Policy actions are applied only to the active motors.
            container.set_motor_cmd_buf[:, container.active_motor_mask] = torch.clamp(
                container.scaled_act_buf[:, container.active_motor_slice], 0.0, None
            )

        # Per-motor passive spring on all motors.
        container.set_motor_cmd_buf[:] += (
            -robot.spring_constants * robot.get_joint_pos_buf[:, robot.motor_to_joint_dof_index]
        )
        gym.set_motor_forces(container.gpu_set_motor_control_command_array)

        # Gravity compensation on base link.
        container.set_force_torque_buf[:, :, 2] = 9.81 * robot.link_masses
        gym.set_link_external_forces(container.set_force_torque_cmd_arr)

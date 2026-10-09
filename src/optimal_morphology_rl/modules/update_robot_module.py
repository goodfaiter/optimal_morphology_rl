"""Module that refreshes robot state buffers after the physics step."""

from __future__ import annotations

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
from optimal_morphology_rl.modules.tendon_scale_utils import build_tendon_scale_tensor


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
        robot.get_tendon_force_buf = torch.zeros((total_num_envs, robot.num_tendons), dtype=torch.float32, device=device)
        robot.calculated_tendon_velocity = torch.zeros((total_num_envs, robot.num_tendons), dtype=torch.float32, device=device)
        robot.calculated_tendon_acceleration = torch.zeros((total_num_envs, robot.num_tendons), dtype=torch.float32, device=device)

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

        get_tendon_vel_cmd = env_group.create_spatial_tendon_state_command(
            v.SpatialTendonState.VELOCITY,
            v.wrap_gpu_buffer(robot.get_tendon_vel_buf),
            robot.arti_handle,
            (0, robot.num_tendons),
        )

        get_tendon_force_cmd = env_group.create_spatial_tendon_state_command(
            v.SpatialTendonState.FORCE,
            v.wrap_gpu_buffer(robot.get_tendon_force_buf),
            robot.arti_handle,
            (0, robot.num_tendons),
        )

        robot.gpu_get_tendon_command_array = gym.create_gpu_array([get_tendon_lengths_cmd, get_tendon_vel_cmd, get_tendon_force_cmd])


def _refresh_buffers(robot: Robot, gym: v.Gym) -> None:
    """Refresh robot kinematic state from simulation."""
    gym.get_articulation_kinematic_states(robot.gpu_get_kinematic_state_command_array)
    if robot.num_distal_links > 0:
        gym.get_link_transforms(robot.gpu_get_distal_link_transforms_cmd_arr)
        robot.distal_link_pos_buf[:] = robot.distal_link_transform_buf[:, :, 4:7]
    if robot.use_tendon:
        gym.get_spatial_tendon_states(robot.gpu_get_tendon_command_array)


def _get_state(container: ModuleContainer) -> dict[str, torch.Tensor]:
    """Update and return robot-derived observation tensors."""
    robot = container.robot

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
        "set_motor_cmd_buf": container.set_motor_cmd_buf,
    }
    if robot.use_tendon:
        state["dof_pos_buf"] = robot.get_tendon_lengths_buf
        state["dof_vel_buf"] = robot.get_tendon_vel_buf
        state["calculated_tendon_velocity"] = robot.calculated_tendon_velocity
        state["calculated_tendon_acceleration"] = robot.calculated_tendon_acceleration
    else:
        state["dof_pos_buf"] = robot.get_joint_pos_buf
        state["dof_vel_buf"] = robot.get_joint_vel_buf
    return state


@register_module("update_robot")
class UpdateRobotModule(BaseModule):
    """Refreshes robot state buffers for the robot created by ``create_robot``.

    For tendon-driven robots the module additionally computes filtered tendon
    velocity and acceleration signals per tendon column from the finite
    difference of the tendon lengths (the simulator's tendon velocity buffer is
    unreliable and unused):

    - ``calculated_tendon_velocity``: the negated length finite difference
      (``-(L - L_prev)/dt`` per step, winding convention ``dL = -radius *
      dtheta`` as in 'tendon_est'), exponentially smoothed (``vel = a*vel_raw +
      (1-a)*vel_prev`` with the ``vel_filter_alpha`` coefficient, 1.0 = raw),
      spike-guarded (``max_tendon_speed`` — on the reset step the buffers still
      show the pre-teleport state, so the next step would otherwise see a
      phantom velocity jump) and soft-debanded (``vel_deadband``, attenuates
      the residual length-jitter noise floor so the acceleration finite
      difference cannot amplify it into bang-bang chatter at near-zero
      velocity)
    - ``calculated_tendon_acceleration``: the finite difference of the velocity
      in tendon units, exponentially smoothed (``acc_filter_alpha`` coefficient,
      1.0 = raw) and soft-debanded (``acc_deadband``)

    Entries without length history (first step after (re)initialization or
    reset) use zero velocity. Both signals are stored on the robot and
    published in ``container.robot_state``; 'tendon_force_m5' consumes them.
    The filter parameters are scalars, or one entry per tendon.
    """

    def post_finalize(self, container: ModuleContainer) -> None:
        """Allocate read buffers, create read GPU commands and set up the tendon dynamics filters."""
        robot = container.robot
        _allocate_read_buffers(robot, container.total_num_envs, container.device)
        _create_read_gpu_commands(robot, container.env_group, container.gym)
        if robot.use_tendon:
            self._setup_tendon_dynamics(container)
        container.robot_state = _get_state(container)

    def _setup_tendon_dynamics(self, container: ModuleContainer) -> None:
        """Build the tendon filter tensors and allocate the length/velocity/acceleration history."""
        robot = container.robot
        device = container.device
        num_tendons = robot.num_tendons

        timestep = container.get("timestep")
        frame_skip = container.get("frame_skip")
        if timestep is None or frame_skip is None:
            raise RuntimeError(
                "update_robot requires 'timestep' and 'frame_skip' in the shared container. "
                "Ensure the 'create_rigid_vsim_envs' module is listed before 'update_robot'."
            )
        self.dt = float(timestep) * float(frame_skip)

        self.max_tendon_speed = build_tendon_scale_tensor(
            self.config.get("max_tendon_speed", 0.22), num_tendons, device, "max_tendon_speed"
        )
        self.vel_filter_alpha = build_tendon_scale_tensor(
            self.config.get("vel_filter_alpha", 1.0), num_tendons, device, "vel_filter_alpha"
        )
        self.acc_filter_alpha = build_tendon_scale_tensor(
            self.config.get("acc_filter_alpha", 1.0), num_tendons, device, "acc_filter_alpha"
        )
        self.vel_deadband = build_tendon_scale_tensor(self.config.get("vel_deadband", 0.0), num_tendons, device, "vel_deadband")
        self.acc_deadband = build_tendon_scale_tensor(self.config.get("acc_deadband", 0.0), num_tendons, device, "acc_deadband")

        if torch.any(self.max_tendon_speed <= 0.0):
            raise RuntimeError("update_robot config 'max_tendon_speed' entries must be positive.")
        if torch.any(self.vel_filter_alpha < 0.0) or torch.any(self.vel_filter_alpha > 1.0):
            raise RuntimeError("update_robot config 'vel_filter_alpha' entries must be in [0, 1].")
        if torch.any(self.acc_filter_alpha < 0.0) or torch.any(self.acc_filter_alpha > 1.0):
            raise RuntimeError("update_robot config 'acc_filter_alpha' entries must be in [0, 1].")
        if torch.any(self.vel_deadband < 0.0):
            raise RuntimeError("update_robot config 'vel_deadband' entries must be non-negative.")
        if torch.any(self.acc_deadband < 0.0):
            raise RuntimeError("update_robot config 'acc_deadband' entries must be non-negative.")

        # Previous filtered tendon velocity per env/tendon (exponential filter state,
        # also the previous sample for the acceleration finite difference).
        self.prev_vel = torch.zeros((container.total_num_envs, num_tendons), device=device, dtype=torch.float32)
        # Previous tendon length per env/tendon for the velocity finite difference.
        # NaN = no history yet: the first step after (re)initialization uses zero velocity.
        self.prev_length = torch.full((container.total_num_envs, num_tendons), float("nan"), device=device, dtype=torch.float32)
        # Previous filtered acceleration per env/tendon (exponential filter state).
        self.prev_acc = torch.zeros((container.total_num_envs, num_tendons), device=device, dtype=torch.float32)

    def _compute_tendon_dynamics(self, robot: Robot) -> None:
        """Compute the filtered tendon velocity/acceleration for every tendon column."""
        tendon_lengths = robot.get_tendon_lengths_buf

        # Finite-difference tendon velocity (the simulator's velocity buffer is unreliable).
        # Entries without length history (first step after reset) use zero velocity.
        has_history = torch.isfinite(self.prev_length)
        vel_raw = -1.0 * (tendon_lengths - self.prev_length) / self.dt
        vel_raw = torch.where(has_history, vel_raw, torch.zeros_like(vel_raw))

        # Spike guard: on the reset step the buffers still show the pre-teleport state,
        # so the first stored length is stale and the next step sees a phantom velocity
        # jump far beyond the physical joint speed limit (which launches the finger).
        # Jumps beyond max_tendon_speed are phantom velocities, not motion.
        vel_raw = torch.where(vel_raw.abs() > self.max_tendon_speed, torch.zeros_like(vel_raw), vel_raw)

        # Exponential smoothing of the noisy finite-difference velocity.
        tendon_vels = self.vel_filter_alpha * vel_raw + (1.0 - self.vel_filter_alpha) * self.prev_vel

        # Soft deadband (continuous, so no FD spikes at the threshold): attenuates the
        # residual length-jitter noise floor so the acceleration finite difference does
        # not amplify it into alternating accelerations (bang-bang force chatter at
        # near-zero velocity).
        tendon_vels = torch.sign(tendon_vels) * torch.clamp(tendon_vels.abs() - self.vel_deadband, min=0.0)

        # Finite-difference acceleration from the filtered velocity (tendon units).
        acc_raw = (tendon_vels - self.prev_vel) / self.dt

        # Exponential smoothing of the noisy finite-difference acceleration.
        acc = self.acc_filter_alpha * acc_raw + (1.0 - self.acc_filter_alpha) * self.prev_acc

        # Soft deadband on the acceleration (same rationale as the velocity deadband).
        acc = torch.sign(acc) * torch.clamp(acc.abs() - self.acc_deadband, min=0.0)

        robot.calculated_tendon_velocity[:] = tendon_vels
        robot.calculated_tendon_acceleration[:] = acc
        self.prev_length[:] = tendon_lengths
        self.prev_vel[:] = tendon_vels
        self.prev_acc[:] = acc

    def step(self, container: ModuleContainer) -> None:
        """Refresh robot state buffers and publish the robot state dict."""
        robot = container.get("robot")
        if robot is None:
            return
        _refresh_buffers(robot, container.gym)
        if robot.use_tendon:
            self._compute_tendon_dynamics(robot)
        container.robot_state = _get_state(container)

    def reset(self, container: ModuleContainer) -> None:
        """Clear the tendon length/velocity/acceleration history of the environments selected by the reset buffer."""
        robot = container.get("robot")
        if robot is None or not robot.use_tendon:
            return
        self.prev_length[container.reset_buf] = float("nan")
        self.prev_vel[container.reset_buf] = 0.0
        self.prev_acc[container.reset_buf] = 0.0

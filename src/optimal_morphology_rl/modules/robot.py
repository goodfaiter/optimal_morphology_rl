from __future__ import annotations

import os
import sys

import torch
import vlearn as v
from vlearn import gym

from optimal_morphology_rl.helpers.numpy_vlearn import quaternion_to_6d

# TODO: Refactor to avoid this hack to import from the vlearn repo.
sys.path.append(os.path.join(os.path.dirname(__file__), "/workspace/vlearn/train/envs/"))
from environment import EnvironmentGpu


class Robot:
    """Helper that owns the hand articulation buffers and GPU commands."""

    def __init__(self, env: EnvironmentGpu):
        self.env: EnvironmentGpu = env
        self.device = self.env.device
        self.gym: gym.Gym = self.env.gym
        self.total_num_envs = self.env.total_num_envs

        self.num_joints = self.env.num_joints
        self.num_links = self.env.num_links
        self.num_motors = self.env.num_motors

        self.gpu_reset_kinematic_state_command_array = None
        self.gpu_set_kinematic_state_command_array = None
        self.gpu_get_kinematic_state_command_array = None
        self.gpu_set_motor_control_command_array = None

    def allocate_buffers(self) -> None:
        """Allocate robot state and control buffers."""

        self.reset_joint_pos_buf = torch.zeros((self.total_num_envs, self.num_joints), device=self.device, dtype=torch.float32)
        self.reset_joint_vel_buf = torch.zeros((self.total_num_envs, self.num_joints), device=self.device, dtype=torch.float32)
        self.reset_root_transform_buf = torch.zeros((self.total_num_envs, 7), device=self.device, dtype=torch.float32)
        self.reset_root_vel_buf = torch.zeros((self.total_num_envs, 6), device=self.device, dtype=torch.float32)

        self.set_joint_pos_buf = torch.zeros((self.total_num_envs, 0), device=self.device, dtype=torch.float32)
        self.set_joint_vel_buf = torch.zeros((self.total_num_envs, 0), device=self.device, dtype=torch.float32)
        self.set_root_transform_buf = torch.zeros((self.total_num_envs, 7), device=self.device, dtype=torch.float32)
        self.set_root_vel_buf = torch.zeros((self.total_num_envs, 6), device=self.device, dtype=torch.float32)

        self.set_motor_cmd_buf = torch.zeros((self.total_num_envs, self.num_joints), device=self.device, dtype=torch.float32)

        self.get_joint_pos_buf = torch.zeros((self.total_num_envs, self.num_joints), device=self.device, dtype=torch.float32)
        self.get_joint_vel_buf = torch.zeros((self.total_num_envs, self.num_joints), device=self.device, dtype=torch.float32)
        self.get_root_transform_buf = torch.zeros((self.total_num_envs, 7), device=self.device, dtype=torch.float32)
        self.get_root_vel_buf = torch.zeros((self.total_num_envs, 6), device=self.device, dtype=torch.float32)

        self.robot_pos_in_world = torch.zeros((self.total_num_envs, 3), device=self.device, dtype=torch.float32)
        self.quat_robot_to_world = torch.zeros((self.total_num_envs, 4), device=self.device, dtype=torch.float32)
        self._6d_robot_to_world = torch.zeros((self.total_num_envs, 6), device=self.device, dtype=torch.float32)
        self.robot_linear_velocity_in_world = torch.zeros((self.total_num_envs, 3), device=self.device, dtype=torch.float32)
        self.robot_angular_velocity_in_world = torch.zeros((self.total_num_envs, 3), device=self.device, dtype=torch.float32)

    def create_gpu_commands(self, env_group, reset_buf: torch.Tensor, inverse_reset_buf: torch.Tensor) -> None:
        """Create GPU commands for robot state and control."""

        reset_kin_cmd = env_group.create_articulation_kinematic_state_command(
            v.wrap_gpu_buffer(self.reset_joint_pos_buf),
            v.wrap_gpu_buffer(self.reset_joint_vel_buf),
            v.wrap_gpu_buffer(self.reset_root_transform_buf),
            v.wrap_gpu_buffer(self.reset_root_vel_buf),
            self.env.arti_handle,
            (0, self.num_joints),
            (0, 1),
            masks_buffer=v.wrap_gpu_buffer(reset_buf),
        )
        self.gpu_reset_kinematic_state_command_array = self.gym.create_gpu_array([reset_kin_cmd])

        set_kin_cmd = env_group.create_articulation_kinematic_state_command(
            v.wrap_gpu_buffer(self.set_joint_pos_buf),
            v.wrap_gpu_buffer(self.set_joint_vel_buf),
            v.wrap_gpu_buffer(self.set_root_transform_buf),
            v.wrap_gpu_buffer(self.set_root_vel_buf),
            self.env.arti_handle,
            (0, 0),
            (0, 1),
            masks_buffer=v.wrap_gpu_buffer(inverse_reset_buf),
        )
        self.gpu_set_kinematic_state_command_array = self.gym.create_gpu_array([set_kin_cmd])

        get_kin_cmd = env_group.create_articulation_kinematic_state_command(
            v.wrap_gpu_buffer(self.get_joint_pos_buf),
            v.wrap_gpu_buffer(self.get_joint_vel_buf),
            v.wrap_gpu_buffer(self.get_root_transform_buf),
            v.wrap_gpu_buffer(self.get_root_vel_buf),
            self.env.arti_handle,
            (0, self.num_joints),
            (0, 1),
        )
        self.gpu_get_kinematic_state_command_array = self.gym.create_gpu_array([get_kin_cmd])

        set_motor_cmd = env_group.create_motor_control_command(
            v.wrap_gpu_buffer(self.set_motor_cmd_buf), self.env.arti_handle, index_range=[0, self.num_motors]
        )
        self.gpu_set_motor_control_command_array = self.gym.create_gpu_array([set_motor_cmd])

    def refresh_buffers(self, gym: v.Gym) -> None:
        """Refresh robot kinematic state from simulation."""
        gym.get_articulation_kinematic_states(self.gpu_get_kinematic_state_command_array)

    def get_state(self) -> dict[str, torch.Tensor]:
        """Update and return the robot-derived observation tensors."""

        self.robot_pos_in_world[:] = self.get_root_transform_buf[:, 4:7]
        self.quat_robot_to_world[:] = self.get_root_transform_buf[:, 0:4]
        self._6d_robot_to_world[:] = quaternion_to_6d(self.quat_robot_to_world)
        self.robot_linear_velocity_in_world[:] = self.get_root_vel_buf[:, 3:6]
        self.robot_angular_velocity_in_world[:] = self.get_root_vel_buf[:, :3]

        return {
            "robot_pos_in_world": self.robot_pos_in_world,
            "quat_robot_to_world": self.quat_robot_to_world,
            "_6d_robot_to_world": self._6d_robot_to_world,
            "robot_linear_velocity_in_world": self.robot_linear_velocity_in_world,
            "robot_angular_velocity_in_world": self.robot_angular_velocity_in_world,
            "get_joint_pos_buf": self.get_joint_pos_buf,
            "get_joint_vel_buf": self.get_joint_vel_buf,
            "get_root_transform_buf": self.get_root_transform_buf,
            "get_root_vel_buf": self.get_root_vel_buf,
            "set_motor_cmd_buf": self.set_motor_cmd_buf,
        }

    def reset(self, reset_buf: torch.Tensor) -> None:
        """Reset robot, object, goal, and episode bookkeeping state."""
        # Reset Hand Kinematics
        # grasp = torch.rand((num_reset, 1), device=self.device) * 0.5 * torch.pi
        # per_finger = torch.ones((num_reset, self.num_joints), device=self.device)
        # self.reset_joint_pos_buf[reset_buf, :] = grasp * per_finger
        # self.reset_joint_pos_buf[reset_buf, :] = 0
        # self.reset_joint_pos_buf[reset_buf, :] = (
        #     torch.rand((num_reset, self.num_joints), device=self.device) * 0.5 * torch.pi
        # )
        self.reset_joint_pos_buf[reset_buf, :] = 0.0
        self.reset_joint_vel_buf[reset_buf, :] = 0.0
        # self.reset_root_transform_buf[reset_buf, :4] = random_uniform_quaternion(num_reset, device=self.device, dtype=torch.float32)
        self.reset_root_transform_buf[reset_buf, :4] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        self.reset_root_transform_buf[reset_buf, 4:] = torch.tensor([[-0.1, -0.15, 0.1]], device=self.device)
        # self.reset_root_transform_buf[reset_buf, 4:] = (
        #     self.env.table_bounds[:, 0]
        #     + 0.1
        #     + torch.rand((num_reset, 3), device=self.device) * (self.env.table_bounds[:, 1] - self.env.table_bounds[:, 0] - 0.1)
        # )
        self.reset_root_vel_buf[reset_buf, :] = 0.0
        self.gym.set_articulation_kinematic_states(self.gpu_reset_kinematic_state_command_array)

        # Randomize rigid body material
        # Write to buffers
        if self.env.total_num_envs > 10: # only randomize if training (more than 10 envs); keep fixed during evaluation for consistency
            friction = torch.rand(len(self.env.num_envs), device=self.device) * 0.95 + 0.05  # [0.05, 1.0]
        else:
            friction = 0.5
        self.env.set_static_friction_buf[:] = friction
        self.env.set_dynamic_friction_buf[:] = friction
        self.gym.set_rigid_material_properties(self.env.gpu_set_friction_cmd)
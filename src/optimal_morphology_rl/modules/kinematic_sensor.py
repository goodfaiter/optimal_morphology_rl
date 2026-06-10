from __future__ import annotations

import os
import sys

import torch
import vlearn as v
from vlearn import gym

# TODO: Refactor to avoid this hack to import from the vlearn repo.
sys.path.append(os.path.join(os.path.dirname(__file__), "/workspace/vlearn/train/envs/"))
from environment import EnvironmentGpu


class KinematicSensor:
    """Helper for reading a kinematic sensor by handle.

    The handle is resolved automatically by trying rigid body first and then
    articulation.
    """

    def __init__(self, env: EnvironmentGpu, handle: int, sensor_index: int = 0):
        self.env: EnvironmentGpu = env
        self.device = self.env.device
        self.gym: gym.Gym = self.env.gym
        self.total_num_envs = self.env.total_num_envs

        env_def = self.gym.get_environment_def(self.env.env_def_handle)
        source = None

        # Prefer rigid body when a handle could refer to multiple source types.
        try:
            source = env_def.get_rigid_body(handle)
        except Exception:
            source = None
        
        if source is None:
            source = env_def.get_articulation(handle)

        if source is None:
            raise ValueError(f"Handle {handle} is neither a rigid body nor an articulation in this environment definition.")

        self.kinematic_sensor_handle = source.get_kinematic_sensor_handle(sensor_index)

        self.pose_buf = torch.zeros((self.total_num_envs, 7), dtype=torch.float32, device=self.device)
        self.velocity_buf = torch.zeros((self.total_num_envs, 6), dtype=torch.float32, device=self.device)

        self.get_kinematic_sensor_cmd = self.env.env_group.create_kinematic_sensor_state_command(
            v.wrap_gpu_buffer(self.pose_buf),
            v.wrap_gpu_buffer(self.velocity_buf),
            self.kinematic_sensor_handle,
            frame_type=v.FrameType.ENVIRONMENT,
        )
        self.get_kinematic_sensor_cmd_arr = self.gym.create_gpu_array([self.get_kinematic_sensor_cmd])

    def update(self) -> None:
        """Read the latest kinematic sensor data into the dense buffers."""
        self.gym.get_kinematic_sensor_states(self.get_kinematic_sensor_cmd_arr)

    @property
    def pose(self) -> torch.Tensor:
        return self.pose_buf

    @property
    def quat_sensor_to_world(self) -> torch.Tensor:
        return self.pose_buf[:, :4]

    @property
    def pos_in_world(self) -> torch.Tensor:
        return self.pose_buf[:, 4:7]

    @property
    def angular_velocity_world(self) -> torch.Tensor:
        return self.velocity_buf[:, :3]

    @property
    def linear_velocity_world(self) -> torch.Tensor:
        return self.velocity_buf[:, 3:6]

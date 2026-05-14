from __future__ import annotations

from typing import Sequence

import os
import sys

import torch
import vlearn as v
from vlearn import gym

# TODO: Refactor to avoid this hack to import from the vlearn repo.
sys.path.append(os.path.join(os.path.dirname(__file__), "/workspace/vlearn/train/envs/"))
from environment import EnvironmentGpu


class ForceSensors:
    """Helper for reading force sensor buffers from an environment."""

    def __init__(self, env: EnvironmentGpu, link_names: Sequence[str] | None = None):
        self.env: EnvironmentGpu = env
        self.device = self.env.device
        self.gym: gym.Gym = self.env.gym
        self.total_num_envs = self.env.total_num_envs

        env_def = self.gym.get_environment_def(self.env.env_def_handle)
        articulation = env_def.get_articulation(self.env.arti_handle)

        link_name_set = None if link_names is None else {name.lower() for name in link_names}

        self.force_sensor_handles = []
        self.force_sensor_link_names = []

        for sensor_index in range(self.env.art_def.get_num_force_sensor_defs()):
            sensor_def = self.env.art_def.get_force_sensor_def(sensor_index)
            sensor_link_name = sensor_def.link_name

            for name in link_name_set:
                if name not in sensor_link_name.lower():
                    continue

            self.force_sensor_handles.append(articulation.get_force_sensor_handle(sensor_index))
            self.force_sensor_link_names.append(sensor_link_name)

        self.num_force_sensors = len(self.force_sensor_handles)
        self.force_sensor_buf = None
        self.get_force_sensor_cmd_arr = None
        self.force_sensor_views = []
        self.force_sensor_cmds = []

        if self.num_force_sensors > 0:
            self.force_sensor_buf = torch.zeros(
                (self.total_num_envs, self.num_force_sensors, 6), dtype=torch.float32, device=self.device
            )
            self.force_sensor_views = [
                torch.zeros((self.total_num_envs, 6), dtype=torch.float32, device=self.device)
                for _ in range(self.num_force_sensors)
            ]

            for sensor_index, force_sensor_handle in enumerate(self.force_sensor_handles):
                self.force_sensor_cmds.append(
                    self.env.env_group.create_force_sensor_command(
                        v.wrap_gpu_buffer(self.force_sensor_views[sensor_index]),
                        force_sensor_handle,
                        frame_type=v.FrameType.ENVIRONMENT,
                    )
                )

            self.get_force_sensor_cmd_arr = self.gym.create_gpu_array(self.force_sensor_cmds)

    def update(self) -> None:
        """Read the latest force sensor data into the dense buffer."""

        if self.get_force_sensor_cmd_arr is None or self.force_sensor_buf is None:
            self.force_sensor_buf.zero_()
            return

        self.gym.get_sensor_forces(self.get_force_sensor_cmd_arr)

        # Not ideal due to for loop but we expected only 4 to 6 sensors per fingertip.
        for sensor_index, sensor_view in enumerate(self.force_sensor_views):
            self.force_sensor_buf[:, sensor_index, :].copy_(sensor_view)
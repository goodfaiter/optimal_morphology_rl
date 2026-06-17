from __future__ import annotations

from typing import Sequence

import torch
import vlearn as v


class ForceSensors:
    """Helper that owns force sensor buffers and GPU commands. Stateless w.r.t. env; callers pass in what's needed."""

    def __init__(self):
        self.num_force_sensors = 0
        self.force_sensor_handles = []
        self.force_sensor_link_names = []
        self.force_sensor_buf = None
        self.get_force_sensor_cmd_arr = None
        self.force_sensor_views = []
        self.force_sensor_cmds = []

    def allocate_buffers(
        self,
        art_def,
        articulation,
        total_num_envs: int,
        device: torch.device,
        link_names: Sequence[str] | None = None,
    ) -> None:
        """Find matching sensors and allocate state buffers."""
        link_name_set = None if link_names is None else {name.lower() for name in link_names}

        for sensor_index in range(art_def.get_num_force_sensor_defs()):
            sensor_def = art_def.get_force_sensor_def(sensor_index)
            sensor_link_name = sensor_def.link_name

            if link_name_set is not None:
                if not any(name in sensor_link_name.lower() for name in link_name_set):
                    continue

            self.force_sensor_handles.append(articulation.get_force_sensor_handle(sensor_index))
            self.force_sensor_link_names.append(sensor_link_name)

        self.num_force_sensors = len(self.force_sensor_handles)

        if self.num_force_sensors > 0:
            self.force_sensor_buf = torch.zeros(
                (total_num_envs, self.num_force_sensors, 6), dtype=torch.float32, device=device
            )
            self.force_sensor_views = [
                torch.zeros((total_num_envs, 6), dtype=torch.float32, device=device)
                for _ in range(self.num_force_sensors)
            ]

    def create_gpu_commands(self, env_group, gym: v.Gym) -> None:
        """Create GPU commands for reading force sensor data."""
        if self.num_force_sensors == 0:
            return

        for force_sensor_handle, sensor_view in zip(self.force_sensor_handles, self.force_sensor_views):
            self.force_sensor_cmds.append(
                env_group.create_force_sensor_command(
                    v.wrap_gpu_buffer(sensor_view),
                    force_sensor_handle,
                    frame_type=v.FrameType.ENVIRONMENT,
                )
            )

        self.get_force_sensor_cmd_arr = gym.create_gpu_array(self.force_sensor_cmds)

    def update(self, gym: v.Gym) -> None:
        """Read the latest force sensor data into the dense buffer."""
        if self.get_force_sensor_cmd_arr is None or self.force_sensor_buf is None:
            return

        gym.get_sensor_forces(self.get_force_sensor_cmd_arr)

        # Not ideal due to for loop but we expect only 4 to 6 sensors per fingertip.
        for sensor_index, sensor_view in enumerate(self.force_sensor_views):
            self.force_sensor_buf[:, sensor_index, :].copy_(sensor_view)

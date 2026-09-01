"""Module that discovers and creates fingertip force sensors on the robot."""

from __future__ import annotations

from typing import Any, Sequence

import torch
import vlearn as v

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


class ForceSensors:
    """Owns force sensor buffers and GPU commands."""

    def __init__(self):
        self.num_force_sensors = 0
        self.force_sensor_handles: list[Any] = []
        self.force_sensor_link_names: list[str] = []
        self.force_sensor_buf: torch.Tensor | None = None
        self.get_force_sensor_cmd_arr: Any = None
        self.force_sensor_views: list[torch.Tensor] = []
        self.force_sensor_cmds: list[Any] = []

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
                (total_num_envs, self.num_force_sensors, 6),
                dtype=torch.float32,
                device=device,
            )
            self.force_sensor_views = [
                torch.zeros((total_num_envs, 6), dtype=torch.float32, device=device) for _ in range(self.num_force_sensors)
            ]

    def create_gpu_commands(self, env_group: Any, gym: v.Gym) -> None:
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
        for sensor_index, sensor_view in enumerate(self.force_sensor_views):
            self.force_sensor_buf[:, sensor_index, :].copy_(sensor_view)


@register_module("create_force_sensor")
class CreateForceSensorModule(BaseModule):
    """Wraps ForceSensors and exposes the buffer on the shared container."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.sensors = ForceSensors()
        self.link_names: Sequence[str] = self.config.get("link_names", ["distal"])

    def finalize(self, container: ModuleContainer) -> None:
        """Verify dependencies."""
        if container.get("robot") is None or container.get("env_def") is None:
            raise RuntimeError("CreateForceSensorModule requires 'robot' and 'env_def' in the shared container.")

    def post_finalize(self, container: ModuleContainer) -> None:
        """Allocate buffers and create GPU commands."""
        env_def = container.env_def
        robot = container.robot
        articulation = env_def.get_articulation(robot.arti_handle)

        self.sensors.allocate_buffers(
            robot.art_def,
            articulation,
            container.total_num_envs,
            container.device,
            link_names=self.link_names,
        )
        container.force_sensors = self.sensors
        self.sensors.create_gpu_commands(container.env_group, container.gym)

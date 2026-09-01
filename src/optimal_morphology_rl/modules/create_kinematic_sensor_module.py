"""Module that creates a kinematic sensor on the reward object."""

from __future__ import annotations

from typing import Any

import torch
import vlearn as v

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


class KinematicSensor:
    """Owns kinematic sensor buffers and GPU commands."""

    def __init__(self):
        self.kinematic_sensor_handle: Any = None
        self.pose_in_world_buf: torch.Tensor | None = None
        self.velocity_in_world_buf: torch.Tensor | None = None
        self.pose_in_object_buf: torch.Tensor | None = None
        self.velocity_in_object_buf: torch.Tensor | None = None
        self.get_kinematic_sensor_cmd_arr: Any = None

    def allocate_buffers(
        self,
        env_def,
        handle: int,
        total_num_envs: int,
        device: torch.device,
        sensor_index: int = 0,
    ) -> None:
        """Resolve the sensor handle and allocate state buffers."""
        source = None
        try:
            source = env_def.get_rigid_body(handle)
        except Exception:
            source = None

        if source is None:
            source = env_def.get_articulation(handle)

        if source is None:
            raise ValueError(f"Handle {handle} is neither a rigid body nor an articulation.")

        self.kinematic_sensor_handle = source.get_kinematic_sensor_handle(sensor_index)

        self.pose_in_world_buf = torch.zeros((total_num_envs, 7), dtype=torch.float32, device=device)
        self.velocity_in_world_buf = torch.zeros((total_num_envs, 6), dtype=torch.float32, device=device)
        self.pose_in_object_buf = torch.zeros((total_num_envs, 7), dtype=torch.float32, device=device)
        self.velocity_in_object_buf = torch.zeros((total_num_envs, 6), dtype=torch.float32, device=device)

    def create_gpu_commands(self, env_group: Any, gym: v.Gym) -> None:
        """Create GPU commands for reading kinematic sensor state."""
        in_world_cmd = env_group.create_kinematic_sensor_state_command(
            v.wrap_gpu_buffer(self.pose_in_world_buf),
            v.wrap_gpu_buffer(self.velocity_in_world_buf),
            self.kinematic_sensor_handle,
            frame_type=v.FrameType.ENVIRONMENT,
        )
        in_object_cmd = env_group.create_kinematic_sensor_state_command(
            v.wrap_gpu_buffer(self.pose_in_object_buf),
            v.wrap_gpu_buffer(self.velocity_in_object_buf),
            self.kinematic_sensor_handle,
            frame_type=v.FrameType.LOCAL,
        )
        self.get_kinematic_sensor_cmd_arr = gym.create_gpu_array([in_world_cmd, in_object_cmd])

    def update(self, gym: v.Gym) -> None:
        """Read the latest kinematic sensor data into the dense buffers."""
        gym.get_kinematic_sensor_states(self.get_kinematic_sensor_cmd_arr)

    @property
    def pose_in_world(self) -> torch.Tensor:
        return self.pose_in_world_buf

    @property
    def quat_sensor_to_world(self) -> torch.Tensor:
        return self.pose_in_world_buf[:, :4]

    @property
    def pos_in_world(self) -> torch.Tensor:
        return self.pose_in_world_buf[:, 4:7]

    @property
    def angular_velocity_world(self) -> torch.Tensor:
        return self.velocity_in_world_buf[:, :3]

    @property
    def linear_velocity_world(self) -> torch.Tensor:
        return self.velocity_in_world_buf[:, 3:6]

    @property
    def pose_in_object(self) -> torch.Tensor:
        return self.pose_in_object_buf

    @property
    def quat_sensor_to_object(self) -> torch.Tensor:
        return self.pose_in_object_buf[:, :4]

    @property
    def pos_in_object(self) -> torch.Tensor:
        return self.pose_in_object_buf[:, 4:7]

    @property
    def angular_velocity_object(self) -> torch.Tensor:
        return self.velocity_in_object_buf[:, :3]

    @property
    def linear_velocity_object(self) -> torch.Tensor:
        return self.velocity_in_object_buf[:, 3:6]


@register_module("create_kinematic_sensor")
class CreateKinematicSensorModule(BaseModule):
    """Creates a kinematic sensor on the reward object."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.sensor = KinematicSensor()

    def finalize(self, container: ModuleContainer) -> None:
        """Verify dependencies."""
        if container.get("reward_object") is None:
            raise RuntimeError("CreateKinematicSensorModule requires 'reward_object' in the shared container.")
        if container.get("env_def") is None:
            raise RuntimeError("CreateKinematicSensorModule requires 'env_def' in the shared container.")

    def post_finalize(self, container: ModuleContainer) -> None:
        """Allocate buffers and create GPU commands."""
        self.sensor.allocate_buffers(
            container.env_def,
            container.reward_object.handle,
            container.total_num_envs,
            container.device,
        )
        container.kinematic_sensor = self.sensor
        self.sensor.create_gpu_commands(container.env_group, container.gym)

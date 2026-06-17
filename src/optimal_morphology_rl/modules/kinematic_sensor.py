from __future__ import annotations

import torch
import vlearn as v


class KinematicSensor:
    """Helper that owns kinematic sensor buffers and GPU commands. Stateless w.r.t. env; callers pass in what's needed.

    The source handle is resolved automatically by trying rigid body first and then articulation.
    """

    def __init__(self):
        self.kinematic_sensor_handle = None
        self.pose_buf = None
        self.velocity_buf = None
        self.get_kinematic_sensor_cmd = None
        self.get_kinematic_sensor_cmd_arr = None

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

        self.pose_buf = torch.zeros((total_num_envs, 7), dtype=torch.float32, device=device)
        self.velocity_buf = torch.zeros((total_num_envs, 6), dtype=torch.float32, device=device)

    def create_gpu_commands(self, env_group, gym: v.Gym) -> None:
        """Create GPU commands for reading kinematic sensor state."""
        self.get_kinematic_sensor_cmd = env_group.create_kinematic_sensor_state_command(
            v.wrap_gpu_buffer(self.pose_buf),
            v.wrap_gpu_buffer(self.velocity_buf),
            self.kinematic_sensor_handle,
            frame_type=v.FrameType.ENVIRONMENT,
        )
        self.get_kinematic_sensor_cmd_arr = gym.create_gpu_array([self.get_kinematic_sensor_cmd])

    def update(self, gym: v.Gym) -> None:
        """Read the latest kinematic sensor data into the dense buffers."""
        gym.get_kinematic_sensor_states(self.get_kinematic_sensor_cmd_arr)

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

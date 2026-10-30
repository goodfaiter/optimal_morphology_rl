"""Module that creates every kinematic sensor defined for the environment."""

from __future__ import annotations

from typing import Any

import torch
import vlearn as v
from vlearn.torch_utils.torch_jit_utils import (
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse,
)

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


class KinematicSensor:
    """Owns kinematic sensor buffers and GPU commands."""

    def __init__(self):
        self.kinematic_sensor_handle: Any = None
        self.name: str = ""
        self.rigid_object: Any = None  # the loaded object of a rigid-body sensor (its angular velocity, see update)
        self.pose_in_world_buf: torch.Tensor | None = None
        self.velocity_in_world_buf: torch.Tensor | None = None
        self.pose_in_object_buf: torch.Tensor | None = None
        self.velocity_in_object_buf: torch.Tensor | None = None
        self.get_kinematic_sensor_cmd_arr: Any = None

        # State expressed in the robot base frame.
        self.pos_in_robot_buf: torch.Tensor | None = None
        self.quat_sensor_to_robot_buf: torch.Tensor | None = None
        self.linear_velocity_in_robot_buf: torch.Tensor | None = None
        self.angular_velocity_in_robot_buf: torch.Tensor | None = None

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

        self.pos_in_robot_buf = torch.zeros((total_num_envs, 3), dtype=torch.float32, device=device)
        self.quat_sensor_to_robot_buf = torch.zeros((total_num_envs, 4), dtype=torch.float32, device=device)
        self.linear_velocity_in_robot_buf = torch.zeros((total_num_envs, 3), dtype=torch.float32, device=device)
        self.angular_velocity_in_robot_buf = torch.zeros((total_num_envs, 3), dtype=torch.float32, device=device)

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

    def update(self, gym: v.Gym, reset_rows: torch.Tensor | None = None) -> None:
        """Read the latest kinematic sensor data into the dense buffers.

        ``reset_rows``: envs reset since the last physics step. vlearn's forward kinematics after a reset updates the
        sensor transforms but not the velocities (they keep the previous episode's), so their velocity is the reset
        state's: the object's (reset) velocity, zero for articulation links (the reset joint velocities are zero).
        """
        gym.get_kinematic_sensor_states(self.get_kinematic_sensor_cmd_arr)
        obj = self.rigid_object
        if obj is not None and obj.get_vel_in_world_buf is not None:
            # vlearn reports the angular velocity of a rigid-body kinematic sensor scaled by sqrt(inertia) (sqrt(I) * w,
            # also in its demo 307 setup; the linear velocity is right but stale after a reset): take the velocity from
            # the object's own kinematic state, read by 'update_objects' (listed before 'update_kinematic_sensor').
            quat = self.pose_in_world_buf[:, :4]
            self.velocity_in_world_buf[:, :3] = obj.angular_velocity_world
            self.velocity_in_world_buf[:, 3:6] = obj.linear_velocity_world
            self.velocity_in_object_buf[:, :3] = quat_rotate_inverse(quat, obj.angular_velocity_world)
            self.velocity_in_object_buf[:, 3:6] = quat_rotate_inverse(quat, obj.linear_velocity_world)
        elif reset_rows is not None:
            self.velocity_in_world_buf[reset_rows] = 0.0
            self.velocity_in_object_buf[reset_rows] = 0.0

    def update_robot_frame(self, robot_state: dict[str, torch.Tensor]) -> None:
        """Express object pose and velocity in the robot base frame.

        Must be called after :meth:`update` and after ``robot_state`` has been
        refreshed by ``update_robot``.
        """
        quat_robot_to_world = robot_state["quat_robot_to_world"]
        robot_pos_in_world = robot_state["robot_pos_in_world"]

        quat_world_to_robot = quat_conjugate(quat_robot_to_world)

        self.pos_in_robot_buf[:] = quat_rotate_inverse(quat_robot_to_world, self.pose_in_world_buf[:, 4:7] - robot_pos_in_world)
        self.quat_sensor_to_robot_buf[:] = quat_mul(quat_world_to_robot, self.pose_in_world_buf[:, :4])

        self.linear_velocity_in_robot_buf[:] = quat_rotate_inverse(quat_robot_to_world, self.velocity_in_world_buf[:, 3:6])
        self.angular_velocity_in_robot_buf[:] = quat_rotate_inverse(quat_robot_to_world, self.velocity_in_world_buf[:, :3])

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

    @property
    def pos_in_robot(self) -> torch.Tensor:
        return self.pos_in_robot_buf

    @property
    def quat_sensor_to_robot(self) -> torch.Tensor:
        return self.quat_sensor_to_robot_buf

    @property
    def linear_velocity_in_robot(self) -> torch.Tensor:
        return self.linear_velocity_in_robot_buf

    @property
    def angular_velocity_in_robot(self) -> torch.Tensor:
        return self.angular_velocity_in_robot_buf


@register_module("create_kinematic_sensor")
class CreateKinematicSensorModule(BaseModule):
    """Creates every kinematic sensor defined for the environment.

    Enumerates the loaded reward object's kinematic sensor (its sensors are
    keyed by the object's body name) and every articulation kinematic sensor
    definition (from the .vsim, keyed by the attachment link name), storing
    them on ``container.kinematic_sensors``. Consumers look sensors up by name
    with :func:`find_kinematic_sensor`.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.sensors: dict[str, KinematicSensor] = {}

    def post_finalize(self, container: ModuleContainer) -> None:
        """Create every available kinematic sensor and register them by name."""
        env_def = container.env_def
        device = container.device

        # 1) The reward object's kinematic sensor (object environments), keyed
        #    by the object's body name.
        reward_object = container.get("reward_object")
        if reward_object is not None:
            sensor = KinematicSensor()
            sensor.allocate_buffers(
                env_def,
                reward_object.handle,
                container.total_num_envs,
                device,
            )
            sensor.name = reward_object.name
            sensor.rigid_object = reward_object
            self.sensors[reward_object.name] = sensor

        # 2) Every articulation kinematic sensor definition (from the .vsim).
        robot = container.get("robot")
        if robot is not None and robot.art_def is not None:
            for i in range(robot.art_def.get_num_kinematic_sensor_defs()):
                sensor = KinematicSensor()
                sensor.allocate_buffers(
                    env_def,
                    robot.arti_handle,
                    container.total_num_envs,
                    device,
                    sensor_index=i,
                )
                sensor.name = robot.art_def.get_kinematic_sensor_def(i).link_name
                self.sensors[sensor.name] = sensor

        container.kinematic_sensors = self.sensors
        for sensor in self.sensors.values():
            sensor.create_gpu_commands(container.env_group, container.gym)


def find_kinematic_sensor(container: ModuleContainer, pattern: str) -> KinematicSensor | None:
    """Look up a created kinematic sensor by body/link name substring."""
    sensors = container.get("kinematic_sensors") or {}
    for key, sensor in sensors.items():
        if pattern.lower() in key.lower():
            return sensor
    return None

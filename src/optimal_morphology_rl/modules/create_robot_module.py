"""Module that loads the robot hand articulation into the environment definition."""

from __future__ import annotations

from typing import Any

import torch
import vlearn as v

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module
from optimal_morphology_rl.helpers.numpy_vlearn import random_uniform_quaternion


class Robot:
    """Robot hand metadata and loaded articulation handles.

    State and control buffers are attached by ``robot_control`` and
    ``update_robot`` during their ``post_finalize`` hooks.
    """

    def __init__(self, fixed_hand: bool = False, use_tendon: bool = False):
        self.use_tendon = use_tendon
        self.fixed_hand = fixed_hand

        self.def_handle: int | None = None
        self.art_def: Any = None
        self.arti_handle: Any = None
        self.num_joints: int | None = None
        self.num_links: int | None = None
        self.num_motors: int | None = None
        self.num_sensors: int | None = None
        self.num_tendons: int | None = None
        self.link_masses: torch.Tensor | None = None
        self.rigid_mat_handle: int | None = None

        self.max_torque: float = 0.1
        self.tendon_max_force: float = 10.0

        # Distal link indices for fingertip-position tracking.
        self.distal_link_indices: list[int] = []
        self.num_distal_links: int = 0

        # Slices used by the control module.
        self.root_slice: slice = slice(0, 0)
        self.active_motor_slice: slice = slice(0, 0)

        # Velocity scaling (filled during load).
        self.velocity_scale: torch.Tensor | None = None
        self.max_velocity: torch.Tensor | None = None
        self.min_active_motor_scale: torch.Tensor | None = None
        self.max_active_motor_scale: torch.Tensor | None = None

        # Motor metadata.
        self.motor_to_joint_dof_index: torch.Tensor | None = None
        self.active_motor_mask: torch.Tensor | None = None
        self.active_motor_indices: torch.Tensor | None = None
        self.num_active_motors: int = 0
        self.spring_constants: torch.Tensor | None = None

        # Buffers and GPU command arrays are attached by control/update modules.
        self.reset_joint_pos_buf: torch.Tensor | None = None
        self.reset_joint_vel_buf: torch.Tensor | None = None
        self.reset_root_transform_buf: torch.Tensor | None = None
        self.reset_root_vel_buf: torch.Tensor | None = None

        self.set_joint_pos_buf: torch.Tensor | None = None
        self.set_joint_vel_buf: torch.Tensor | None = None
        self.set_root_transform_buf: torch.Tensor | None = None
        self.set_root_vel_buf: torch.Tensor | None = None

        self.set_motor_cmd_buf: torch.Tensor | None = None
        self.set_tendon_controls_buf: torch.Tensor | None = None
        self.set_force_torque_buf: torch.Tensor | None = None
        self.set_static_friction_buf: torch.Tensor | None = None
        self.set_dynamic_friction_buf: torch.Tensor | None = None

        self.get_joint_pos_buf: torch.Tensor | None = None
        self.get_joint_vel_buf: torch.Tensor | None = None
        self.get_root_transform_buf: torch.Tensor | None = None
        self.get_root_vel_buf: torch.Tensor | None = None
        self.get_tendon_lengths_buf: torch.Tensor | None = None
        self.get_tendon_vel_buf: torch.Tensor | None = None

        self.distal_link_transform_buf: torch.Tensor | None = None
        self.distal_link_pos_buf: torch.Tensor | None = None

        self.scaled_act_buf: torch.Tensor | None = None

        self.robot_pos_in_world: torch.Tensor | None = None
        self.quat_robot_to_world: torch.Tensor | None = None
        self._6d_robot_to_world: torch.Tensor | None = None
        self.robot_linear_velocity_in_world: torch.Tensor | None = None
        self.robot_angular_velocity_in_world: torch.Tensor | None = None
        self.gravity_direction_world: torch.Tensor | None = None
        self.gravity_vector_in_robot_frame: torch.Tensor | None = None
        self.robot_linear_velocity_in_robot_frame: torch.Tensor | None = None
        self.robot_angular_velocity_in_robot_frame: torch.Tensor | None = None

        self.gpu_reset_kinematic_state_command_array: Any = None
        self.gpu_set_kinematic_state_command_array: Any = None
        self.gpu_get_kinematic_state_command_array: Any = None
        self.gpu_get_distal_link_transforms_cmd_arr: Any = None
        self.gpu_set_motor_control_command_array: Any = None
        self.gpu_set_tendon_control_command_array: Any = None
        self.gpu_get_tendon_lengths_command_array: Any = None
        self.gpu_get_tendon_velocities_command_array: Any = None
        self.set_force_torque_cmd_arr: Any = None
        self.gpu_set_friction_cmd: Any = None

    def create_envs(self, env_def, vsim_path: str, device: torch.device) -> None:
        """Load the hand model into the environment definition."""
        print(f"Loading hand model from {vsim_path}")

        env_def.import_definitions(
            vsim_path,
            fixed=self.fixed_hand,
            use_visual_mesh=True,
            merge_fixed_joints=True,
            force_mass_computation=False,
            force_inertia_computation=False,
            query_mode=v.QueryMode.USE_COLLISIONS,
        )

        self.def_handle = env_def.get_articulation_def_handle(0)
        self.art_def = env_def.get_articulation_def(self.def_handle)
        self.art_def.has_self_collisions = False
        self.art_def.enable_control_type(v.ArticulationControlType.MOTOR, True)

        self.arti_handle = env_def.create_articulation(
            self.def_handle,
            v.Transform(v.Quat(0, 0, 0, 1), v.Vec3(0, 0, 0)),
            "hand",
        )

        self.num_joints = self.art_def.get_num_joint_dof_defs()
        self.num_links = self.art_def.get_num_link_defs()
        self.num_motors = self.art_def.get_num_motor_defs()
        self.num_sensors = self.art_def.get_num_force_sensor_defs()
        if self.use_tendon:
            self.num_tendons = self.art_def.get_num_spatial_tendon_defs()

        self.distal_link_indices = [i for i in range(self.num_links) if self.art_def.get_link_def(i).name.lower().endswith("distal")]
        self.num_distal_links = len(self.distal_link_indices)

        self.link_masses = torch.zeros(self.num_links, dtype=torch.float32, device=device)
        for i in range(self.num_links):
            link_def = self.art_def.get_link_def(i)
            self.link_masses[i] = link_def.mass

        self.motor_to_joint_dof_index = torch.zeros(
            self.num_motors, dtype=torch.long, device=device
        )
        for i in range(self.num_motors):
            motor_def = self.art_def.get_motor_def(i)
            self.motor_to_joint_dof_index[i] = motor_def.dof_index

        for i in range(self.num_joints):
            print(self.art_def.get_joint_def(i))

        for i in range(self.num_links):
            print(self.art_def.get_link_def(i))

        for i in range(self.num_motors):
            print(i, self.art_def.get_motor_def(i))

        for i in range(self.num_sensors):
            print(i, self.art_def.get_force_sensor_def(i))

        rigid_mat = v.RigidMaterial()
        rigid_mat.dynamic_friction = 0.5
        rigid_mat.static_friction = 0.5
        rigid_mat.restitution = 0.0
        rigid_mat.damping = 0.0
        rigid_mat_handle = env_def.create_rigid_material(rigid_mat)
        for i in range(self.art_def.get_num_link_defs()):
            env_def.assign_rigid_material_to_articulation_link(self.def_handle, rigid_mat_handle, i)
        self.rigid_mat_handle = rigid_mat_handle

        self.velocity_scale = torch.tensor([1.0, 1.0, 1.0, 0.2, 0.2, 0.2], dtype=torch.float32, device=device)
        self.max_velocity = self.velocity_scale * 2.0

        # Per-motor passive spring constants. All motors get the same constant for now.
        self.spring_constants = torch.full((self.num_motors,), 0.1, dtype=torch.float32, device=device)

        # Active motor mask must be built from config after create_envs.
        self.build_active_motor_mask({})

    def build_active_motor_mask(self, config: dict[str, Any]) -> None:
        """Build the mask selecting which motors receive policy actions.

        By default motors whose names contain any of the substrings in
        ``passive_motor_substrings`` are passive (spring only); everything else
        is active. Alternatively ``active_motor_substrings`` can be provided to
        explicitly select active motors.
        """
        if self.art_def is None or self.num_motors is None:
            raise RuntimeError("create_envs must be called before build_active_motor_mask")

        device = self.motor_to_joint_dof_index.device

        active_substrings = config.get("active_motor_substrings")
        passive_substrings = config.get("passive_motor_substrings", ["abd"])

        mask = torch.ones(self.num_motors, dtype=torch.bool, device=device)
        for i in range(self.num_motors):
            name = self.art_def.get_motor_def(i).name.lower()
            if active_substrings is not None:
                mask[i] = any(sub.lower() in name for sub in active_substrings)
            else:
                mask[i] = not any(sub.lower() in name for sub in passive_substrings)

        self.active_motor_mask = mask
        self.active_motor_indices = torch.nonzero(mask, as_tuple=False).flatten()
        self.num_active_motors = int(mask.sum().item())

        min_scale = -1.0 * self.max_torque
        max_scale = 1.0 * self.max_torque
        if self.use_tendon:
            min_scale = -0.25 * self.tendon_max_force
            max_scale = 1.0 * self.tendon_max_force

        self.min_active_motor_scale = torch.full((self.num_active_motors,), min_scale, device=device)
        self.max_active_motor_scale = torch.full((self.num_active_motors,), max_scale, device=device)

        self.root_slice = slice(0, 6) if not self.fixed_hand else slice(0, 0)
        if self.fixed_hand:
            self.active_motor_slice = slice(0, self.num_active_motors)
        else:
            self.active_motor_slice = slice(6, 6 + self.num_active_motors)

    def get_num_dofs(self) -> int:
        """Return the number of degrees of freedom (joints) in the robot."""
        return self.num_tendons if self.use_tendon else self.num_joints

    def get_num_active_motors(self) -> int:
        """Return the number of motors that receive policy actions."""
        return self.num_tendons if self.use_tendon else self.num_active_motors

    def get_num_actions(self) -> int:
        """Return the number of actions for the robot."""
        return self.get_num_active_motors() if self.fixed_hand else 6 + self.get_num_active_motors()


@register_module("create_robot")
class RobotModule(BaseModule):
    """Loads and exposes the robot hand.

    Expects the shared container to already contain ``env_def`` and ``device``
    (populated by ``create_rigid_vsim_envs``).
    """

    def finalize(self, container: ModuleContainer) -> None:
        """Load the robot hand into the environment definition."""
        if container.get("env_def") is None:
            raise RuntimeError(
                "RobotModule requires 'env_def' in the shared container. Ensure create_rigid_vsim_envs is listed before create_robot."
            )
        if container.get("device") is None:
            raise RuntimeError("RobotModule requires 'device' in the shared container.")

        vsim_path = self.config.get("vsim_path")
        if vsim_path is None:
            raise ValueError("create_robot config missing 'vsim_path'")

        self.fixed_hand = bool(self.config.get("fixed_hand", False))
        self.use_tendon = bool(self.config.get("use_tendon", True))

        self.robot = Robot(fixed_hand=self.fixed_hand, use_tendon=self.use_tendon)
        self.robot.create_envs(container.env_def, vsim_path, container.device)
        self.robot.build_active_motor_mask(self.config)

        self.randomize_pose = bool(self.config.get("randomize_pose", False))
        self.fric_coeff = self.config.get("friction_coefficient", None)

        default_quat = [0.6963642, 0.1227878, -0.1227878, 0.6963642]
        quat = self.config.get("fixed_hand_root_quat", default_quat)
        if not (isinstance(quat, (list, tuple)) and len(quat) == 4):
            raise ValueError("fixed_hand_root_quat must be a list/tuple of 4 floats")
        self.fixed_hand_root_quat = torch.tensor(quat, dtype=torch.float32, device=container.device)

        default_pos = [-0.1, -0.15, 0.1]
        pos = self.config.get("fixed_hand_root_pos", default_pos)
        if not (isinstance(pos, (list, tuple)) and len(pos) == 3):
            raise ValueError("fixed_hand_root_pos must be a list/tuple of 3 floats")
        self.fixed_hand_root_pos = torch.tensor(pos, dtype=torch.float32, device=container.device)

        container.robot = self.robot
        container.robot_vsim_path = vsim_path

    def reset(self, container: ModuleContainer) -> None:
        """Reset the robot hand to its initial state."""
        robot = container.robot
        reset_buf = container.reset_buf
        device = container.device
        gym = container.gym

        randomize_pose = bool(self.config.get("randomize_pose", False))
        fric_coeff = self.config.get("friction_coefficient", None)

        robot.reset_joint_pos_buf[reset_buf, :] = 0.0
        robot.reset_joint_vel_buf[reset_buf, :] = 0.0
        if self.fixed_hand:
            robot.reset_root_transform_buf[reset_buf, 4:] = self.fixed_hand_root_pos
            robot.reset_root_transform_buf[reset_buf, :4] = self.fixed_hand_root_quat
        else:
            if randomize_pose:
                n_reset = reset_buf.sum().item()
                robot.reset_root_transform_buf[reset_buf, :4] = random_uniform_quaternion(n_reset, device=device, dtype=torch.float32)
                robot.reset_root_transform_buf[reset_buf, 4] = -0.1
                robot.reset_root_transform_buf[reset_buf, 5] = torch.rand(n_reset, device=device) * 0.3 - 0.15
                robot.reset_root_transform_buf[reset_buf, 6] = torch.rand(n_reset, device=device) * 0.2 + 0.1
            else:
                robot.reset_root_transform_buf[reset_buf, 4:] = torch.tensor([[-0.1, -0.15, 0.2]], device=device)
                robot.reset_root_transform_buf[reset_buf, :4] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
        robot.reset_root_vel_buf[reset_buf, :] = 0.0
        gym.set_articulation_kinematic_states(robot.gpu_reset_kinematic_state_command_array)

        total_num_envs = reset_buf.shape[0]
        if total_num_envs != 1 and fric_coeff is None:
            static_friction = torch.rand(1, device=device).item() * 0.9 + 0.1
        else:
            static_friction = 0.1 if fric_coeff is None else fric_coeff
        dynamic_friction = static_friction * 0.75

        robot.set_static_friction_buf[0] = static_friction * 2.0
        robot.set_dynamic_friction_buf[0] = dynamic_friction * 2.0
        gym.set_rigid_material_properties(robot.gpu_set_friction_cmd)

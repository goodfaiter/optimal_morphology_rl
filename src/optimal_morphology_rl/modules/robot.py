from __future__ import annotations

import torch
import vlearn as v

from vlearn.torch_utils.torch_jit_utils import scale

from optimal_morphology_rl.helpers.numpy_vlearn import quaternion_to_6d


class Robot:
    def __init__(self, fixed_hand: bool = False, use_tendon: bool = False):
        self.use_tendon = use_tendon
        self.fixed_hand = fixed_hand

        self.gpu_reset_kinematic_state_command_array = None
        self.gpu_set_kinematic_state_command_array = None
        self.gpu_get_kinematic_state_command_array = None

        self.def_handle = None
        self.art_def = None
        self.arti_handle = None
        self.num_joints = None
        self.num_links = None
        self.num_motors = None
        self.num_sensors = None
        self.link_masses = None
        self.rigid_mat_handle = None

        self.max_torque = 0.1
        self.gpu_set_motor_control_command_array = None

        self.num_tendons = None
        self.tendon_max_force = 10.0
        self.gpu_set_tendon_control_command_array = None
        self.gpu_get_tendon_lengths_command_array = None
        self.gpu_get_tendon_velocities_command_array = None

    def allocate_buffers(self, total_num_envs: int, device: torch.device) -> None:
        """Allocate robot state and control buffers."""
        self.reset_joint_pos_buf = torch.zeros((total_num_envs, self.num_joints), device=device, dtype=torch.float32)
        self.reset_joint_vel_buf = torch.zeros((total_num_envs, self.num_joints), device=device, dtype=torch.float32)
        self.reset_root_transform_buf = torch.zeros((total_num_envs, 7), device=device, dtype=torch.float32)
        self.reset_root_vel_buf = torch.zeros((total_num_envs, 6), device=device, dtype=torch.float32)

        self.set_joint_pos_buf = torch.zeros((total_num_envs, 0), device=device, dtype=torch.float32)
        self.set_joint_vel_buf = torch.zeros((total_num_envs, 0), device=device, dtype=torch.float32)
        self.set_root_transform_buf = torch.zeros((total_num_envs, 7), device=device, dtype=torch.float32)
        self.set_root_vel_buf = torch.zeros((total_num_envs, 6), device=device, dtype=torch.float32)

        self.set_motor_cmd_buf = torch.zeros((total_num_envs, self.num_motors), device=device, dtype=torch.float32)

        self.get_joint_pos_buf = torch.zeros((total_num_envs, self.num_joints), device=device, dtype=torch.float32)
        self.get_joint_vel_buf = torch.zeros((total_num_envs, self.num_joints), device=device, dtype=torch.float32)
        self.get_root_transform_buf = torch.zeros((total_num_envs, 7), device=device, dtype=torch.float32)
        self.get_root_vel_buf = torch.zeros((total_num_envs, 6), device=device, dtype=torch.float32)

        self.robot_pos_in_world = torch.zeros((total_num_envs, 3), device=device, dtype=torch.float32)
        self.quat_robot_to_world = torch.zeros((total_num_envs, 4), device=device, dtype=torch.float32)
        self._6d_robot_to_world = torch.zeros((total_num_envs, 6), device=device, dtype=torch.float32)
        self.robot_linear_velocity_in_world = torch.zeros((total_num_envs, 3), device=device, dtype=torch.float32)
        self.robot_angular_velocity_in_world = torch.zeros((total_num_envs, 3), device=device, dtype=torch.float32)

        self.set_force_torque_buf = torch.zeros((total_num_envs, self.num_links, 6), dtype=torch.float32, device=device)

        # Rigid Material Buffers
        self.set_static_friction_buf = torch.zeros(total_num_envs, dtype=torch.float32, device=device)
        self.set_dynamic_friction_buf = torch.zeros(total_num_envs, dtype=torch.float32, device=device)

        if self.use_tendon:
            self.set_tendon_controls_buf = torch.zeros((total_num_envs, self.num_tendons), dtype=torch.float32, device=device)
            self.get_tendon_lengths_buf = torch.zeros((total_num_envs, self.num_tendons), dtype=torch.float32, device=device)
            self.get_tendon_vel_buf = torch.zeros((total_num_envs, self.num_tendons), dtype=torch.float32, device=device)

        self.scaled_act_buf = torch.zeros((total_num_envs, self.get_num_actions()), dtype=torch.float32, device=device)

    def create_envs(self, env_def, vsim_path: str, device: torch.device):
        """Load the hand model into the environment definition and create its articulation."""
        print(f"Loading hand model from {vsim_path}")

        env_def.import_definitions(
            vsim_path,
            fixed=self.fixed_hand,
            use_visual_mesh=False,
            merge_fixed_joints=True,
            force_mass_computation=False,
            force_inertia_computation=False,
            query_mode=v.QueryMode.USE_COLLISIONS,
        )

        self.def_handle = env_def.get_articulation_def_handle(0)
        self.art_def = env_def.get_articulation_def(self.def_handle)
        self.art_def.has_self_collisions = False
        self.art_def.enable_control_type(v.ArticulationControlType.MOTOR, True)

        self.arti_handle = env_def.create_articulation(self.def_handle, v.Transform(v.Quat(0, 0, 0, 1), v.Vec3(0, 0, 0)), "hand")

        self.num_joints = self.art_def.get_num_joint_dof_defs()
        self.num_links = self.art_def.get_num_link_defs()
        self.num_motors = self.art_def.get_num_motor_defs()
        self.num_sensors = self.art_def.get_num_force_sensor_defs()
        if self.use_tendon:
            self.num_tendons = self.art_def.get_num_spatial_tendon_defs()
        self.link_masses = torch.zeros(self.num_links, dtype=torch.float32, device=device)
        for i in range(self.num_links):
            link_def = self.art_def.get_link_def(i)
            self.link_masses[i] = link_def.mass

        for i in range(self.num_joints):
            print(self.art_def.get_joint_def(i))

        for i in range(self.num_links):
            print(self.art_def.get_link_def(i))

        for i in range(self.num_joints):
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

        min_scale = -1.0 * self.max_torque
        max_scale = 1.0 * self.max_torque
        if self.use_tendon:
            min_scale = -0.25 * self.tendon_max_force
            max_scale = 1.0 * self.tendon_max_force

        self.min_revolute_scale = torch.full((self.get_num_dofs(),), min_scale, device=device)
        self.max_revolute_scale = torch.full((self.get_num_dofs(),), max_scale, device=device)

        self.root_slice = slice(0, 6) if not self.fixed_hand else slice(0, 0)
        self.dof_slice = slice(0, self.get_num_dofs()) if self.fixed_hand else slice(6, 6 + self.get_num_dofs())

    def create_gpu_commands(self, env_group, gym: v.Gym, reset_buf: torch.Tensor, inverse_reset_buf: torch.Tensor) -> None:
        """Create GPU commands for robot state and control."""

        reset_kin_cmd = env_group.create_articulation_kinematic_state_command(
            v.wrap_gpu_buffer(self.reset_joint_pos_buf),
            v.wrap_gpu_buffer(self.reset_joint_vel_buf),
            v.wrap_gpu_buffer(self.reset_root_transform_buf),
            v.wrap_gpu_buffer(self.reset_root_vel_buf),
            self.arti_handle,
            (0, self.num_joints),
            (0, 1),
            masks_buffer=v.wrap_gpu_buffer(reset_buf),
        )
        self.gpu_reset_kinematic_state_command_array = gym.create_gpu_array([reset_kin_cmd])

        set_kin_cmd = env_group.create_articulation_kinematic_state_command(
            v.wrap_gpu_buffer(self.set_joint_pos_buf),
            v.wrap_gpu_buffer(self.set_joint_vel_buf),
            v.wrap_gpu_buffer(self.set_root_transform_buf),
            v.wrap_gpu_buffer(self.set_root_vel_buf),
            self.arti_handle,
            (0, 0),
            (0, 1),
            masks_buffer=v.wrap_gpu_buffer(inverse_reset_buf),
        )
        self.gpu_set_kinematic_state_command_array = gym.create_gpu_array([set_kin_cmd])

        get_kin_cmd = env_group.create_articulation_kinematic_state_command(
            v.wrap_gpu_buffer(self.get_joint_pos_buf),
            v.wrap_gpu_buffer(self.get_joint_vel_buf),
            v.wrap_gpu_buffer(self.get_root_transform_buf),
            v.wrap_gpu_buffer(self.get_root_vel_buf),
            self.arti_handle,
            (0, self.num_joints),
            (0, 1),
        )
        self.gpu_get_kinematic_state_command_array = gym.create_gpu_array([get_kin_cmd])

        set_motor_cmd = env_group.create_motor_control_command(
            v.wrap_gpu_buffer(self.set_motor_cmd_buf), self.arti_handle, index_range=[0, self.num_motors]
        )
        self.gpu_set_motor_control_command_array = gym.create_gpu_array([set_motor_cmd])

        if self.use_tendon:
            set_tendon_cmd = env_group.create_spatial_tendon_control_command(
                v.wrap_gpu_buffer(self.set_tendon_controls_buf), self.arti_handle
            )
            self.gpu_set_tendon_control_command_array = gym.create_gpu_array([set_tendon_cmd])

            get_tendon_lengths_cmd = env_group.create_spatial_tendon_state_command(
                v.SpatialTendonState.LENGTH, v.wrap_gpu_buffer(self.get_tendon_lengths_buf), self.arti_handle, (0, self.num_tendons)
            )
            self.gpu_get_tendon_lengths_command_array = gym.create_gpu_array([get_tendon_lengths_cmd])

            get_tendon_vel_cmd = env_group.create_spatial_tendon_state_command(
                v.SpatialTendonState.VELOCITY, v.wrap_gpu_buffer(self.get_tendon_vel_buf), self.arti_handle, (0, self.num_tendons)
            )
            self.gpu_get_tendon_velocities_command_array = gym.create_gpu_array([get_tendon_vel_cmd])

        ### Gravity Comp Commands
        # Create external force command
        set_force_torque_cmd = env_group.create_link_external_force_command(
            v.wrap_gpu_buffer(self.set_force_torque_buf), self.arti_handle, [0, self.num_links], force_type=v.ForceType.FORCE_TORQUE
        )

        self.set_force_torque_cmd_arr = gym.create_gpu_array([set_force_torque_cmd])

        ### Rigid Material Commands
        set_static_friction_cmd = env_group.create_rigid_material_property_command(
            v.RigidMaterialProperty.STATIC_FRICTION,
            v.wrap_gpu_buffer(self.set_static_friction_buf),
            self.rigid_mat_handle,
            v.wrap_gpu_buffer(reset_buf),
        )
        set_dynamic_friction_cmd = env_group.create_rigid_material_property_command(
            v.RigidMaterialProperty.DYNAMIC_FRICTION,
            v.wrap_gpu_buffer(self.set_dynamic_friction_buf),
            self.rigid_mat_handle,
            v.wrap_gpu_buffer(reset_buf),
        )
        self.gpu_set_friction_cmd = gym.create_gpu_array([set_static_friction_cmd, set_dynamic_friction_cmd])

    def refresh_buffers(self, gym: v.Gym) -> None:
        """Refresh robot kinematic state from simulation."""
        gym.get_articulation_kinematic_states(self.gpu_get_kinematic_state_command_array)
        if self.use_tendon:
            gym.get_spatial_tendon_states(self.gpu_get_tendon_lengths_command_array)
            gym.get_spatial_tendon_states(self.gpu_get_tendon_velocities_command_array)

    def get_num_dofs(self) -> int:
        """Return the number of degrees of freedom (joints) in the robot."""
        return self.num_tendons if self.use_tendon else self.num_motors

    def get_num_actions(self) -> int:
        """Return the number of actions for the robot."""
        return self.get_num_dofs() if self.fixed_hand else 6 + self.get_num_dofs()

    def get_state(self) -> dict[str, torch.Tensor]:
        """Update and return the robot-derived observation tensors."""

        self.robot_pos_in_world[:] = self.get_root_transform_buf[:, 4:7]
        self.quat_robot_to_world[:] = self.get_root_transform_buf[:, 0:4]
        self._6d_robot_to_world[:] = quaternion_to_6d(self.quat_robot_to_world)
        self.robot_linear_velocity_in_world[:] = self.get_root_vel_buf[:, 3:6]
        self.robot_angular_velocity_in_world[:] = self.get_root_vel_buf[:, :3]

        state = {
            "robot_pos_in_world": self.robot_pos_in_world,
            "quat_robot_to_world": self.quat_robot_to_world,
            "_6d_robot_to_world": self._6d_robot_to_world,
            "robot_linear_velocity_in_world": self.robot_linear_velocity_in_world,
            "robot_angular_velocity_in_world": self.robot_angular_velocity_in_world,
            "get_root_transform_buf": self.get_root_transform_buf,
            "get_root_vel_buf": self.get_root_vel_buf,
            "set_motor_cmd_buf": self.set_motor_cmd_buf,
        }
        if self.use_tendon:
            state["dof_pos_buf"] = self.get_tendon_lengths_buf
            state["dof_vel_buf"] = self.get_tendon_vel_buf
        else:
            state["dof_pos_buf"] = self.get_joint_pos_buf
            state["dof_vel_buf"] = self.get_joint_vel_buf
        return state

    def reset_idx(self, gym: v.Gym, reset_buf: torch.Tensor, device: torch.device) -> None:
        """Reset robot kinematic state for the given reset indices."""
        self.reset_joint_pos_buf[reset_buf, :] = 0.0
        self.reset_joint_vel_buf[reset_buf, :] = 0.0
        if self.fixed_hand:
            self.reset_root_transform_buf[reset_buf, :4] = torch.tensor([0.7, 0.0, 0.0, 0.7], device=device)
        else:
            self.reset_root_transform_buf[reset_buf, :4] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
        self.reset_root_transform_buf[reset_buf, 4:] = torch.tensor([[-0.1, -0.15, 0.1]], device=device)
        self.reset_root_vel_buf[reset_buf, :] = 0.0
        gym.set_articulation_kinematic_states(self.gpu_reset_kinematic_state_command_array)

        # Randomize rigid body material
        total_num_envs = reset_buf.shape[0]
        if total_num_envs != 1:
            static_friction = torch.rand(reset_buf.sum().item(), device=device) * 0.9 + 0.1
            dynamic_friction = static_friction * 0.75
        else:
            static_friction = 0.1
            dynamic_friction = static_friction * 0.75

        # The friction is average between two objects. So we set object friction to 0 and the robot hand to desired * 2
        static_friction = static_friction * 2.0
        dynamic_friction = dynamic_friction * 2.0

        self.set_static_friction_buf[reset_buf] = static_friction
        self.set_dynamic_friction_buf[reset_buf] = dynamic_friction
        gym.set_rigid_material_properties(self.gpu_set_friction_cmd)

    def pre_physics_step(
        self,
        gym: v.Gym,
        act_buf: torch.Tensor,
    ) -> None:
        """Apply wrist velocity, joint motor commands, and gravity compensation."""
        self.scaled_act_buf[:, self.root_slice] = scale(
            act_buf[:, self.root_slice], -self.velocity_scale[self.root_slice], self.velocity_scale[self.root_slice]
        )
        self.scaled_act_buf[:, self.dof_slice] = scale(act_buf[:, self.dof_slice], self.min_revolute_scale, self.max_revolute_scale)

        # Apply wrist velocity commands
        if not self.fixed_hand:
            self.set_root_transform_buf[:] = self.get_root_transform_buf
            self.set_root_vel_buf[:] = torch.clamp(self.scaled_act_buf[:, self.root_slice], -self.max_velocity, self.max_velocity)
            gym.set_articulation_kinematic_states(self.gpu_set_kinematic_state_command_array)

        self.set_motor_cmd_buf[:] = 0.0

        if self.use_tendon:
            # Apply tendon forces directly from the action
            self.set_tendon_controls_buf[:] = torch.clamp(self.scaled_act_buf[:, self.dof_slice], 0.0, None)
            gym.set_spatial_tendon_forces(self.gpu_set_tendon_control_command_array)
        else:
            # Apply joint motor commands
            self.set_motor_cmd_buf[:] = torch.clamp(self.scaled_act_buf[:, self.dof_slice], 0.0, None)

        # Apply anatgonistic spring to all joints
        self.set_motor_cmd_buf[:] += -0.1 * self.get_joint_pos_buf

        gym.set_motor_forces(self.gpu_set_motor_control_command_array)

        # Gravity compensation on base link
        self.set_force_torque_buf[:, :, 2] = 9.81 * self.link_masses
        gym.set_link_external_forces(self.set_force_torque_cmd_arr)

    def pre_gym_step(self, gym):

        # Apply wrist velocity commands
        if not self.fixed_hand:
            self.set_root_transform_buf[:] = self.get_root_transform_buf
            self.set_root_vel_buf[:] = torch.clamp(self.scaled_act_buf[:, self.root_slice], -self.max_velocity, self.max_velocity)
            gym.set_articulation_kinematic_states(self.gpu_set_kinematic_state_command_array)

        self.set_motor_cmd_buf[:] = 0.0

        if self.use_tendon:
            # Apply tendon forces directly from the action
            self.set_tendon_controls_buf[:] = torch.clamp(self.scaled_act_buf[:, self.dof_slice], 0.0, None)
            gym.set_spatial_tendon_forces(self.gpu_set_tendon_control_command_array)
        else:
            # Apply joint motor commands
            self.set_motor_cmd_buf[:] = torch.clamp(self.scaled_act_buf[:, self.dof_slice], 0.0, None)

        # Apply anatgonistic spring to all joints
        self.set_motor_cmd_buf[:] += -0.1 * self.get_joint_pos_buf

        gym.set_motor_forces(self.gpu_set_motor_control_command_array)

        # Gravity compensation on base link
        self.set_force_torque_buf[:, :, 2] = 9.81 * self.link_masses
        gym.set_link_external_forces(self.set_force_torque_cmd_arr)

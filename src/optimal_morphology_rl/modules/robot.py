from __future__ import annotations

import torch
import vlearn as v

from optimal_morphology_rl.helpers.numpy_vlearn import quaternion_to_6d


class Robot:
    """Helper that owns the hand articulation buffers and GPU commands. Stateless w.r.t. env; callers pass in what's needed."""

    def __init__(self):
        self.gpu_reset_kinematic_state_command_array = None
        self.gpu_set_kinematic_state_command_array = None
        self.gpu_get_kinematic_state_command_array = None
        self.gpu_set_motor_control_command_array = None

        self.def_handle = None
        self.art_def = None
        self.arti_handle = None
        self.num_joints = None
        self.num_links = None
        self.num_motors = None
        self.num_sensors = None
        self.link_masses = None
        self.rigid_mat_handle = None

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

        self.set_motor_cmd_buf = torch.zeros((total_num_envs, self.num_joints), device=device, dtype=torch.float32)

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

    def create_envs(self, env_def, fixed_hand: bool, vsim_path: str, device: torch.device):
        """Load the hand model into the environment definition and create its articulation."""
        print(f"Loading hand model from {vsim_path}")

        env_def.import_definitions(
            vsim_path,
            fixed=fixed_hand,
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
        rigid_mat_handle = env_def.create_rigid_material(rigid_mat)
        for i in range(self.art_def.get_num_link_defs()):
            env_def.assign_rigid_material_to_articulation_link(self.def_handle, rigid_mat_handle, i)
        self.rigid_mat_handle = rigid_mat_handle

    def create_gpu_commands(
        self, env_group, gym: v.Gym, reset_buf: torch.Tensor, inverse_reset_buf: torch.Tensor
    ) -> None:
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

    def reset_idx(self, gym: v.Gym, reset_buf: torch.Tensor, device: torch.device) -> None:
        """Reset robot kinematic state for the given reset indices."""
        self.reset_joint_pos_buf[reset_buf, :] = 0.0
        self.reset_joint_vel_buf[reset_buf, :] = 0.0
        self.reset_root_transform_buf[reset_buf, :4] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
        self.reset_root_transform_buf[reset_buf, 4:] = torch.tensor([[-0.1, -0.15, 0.1]], device=device)
        self.reset_root_vel_buf[reset_buf, :] = 0.0
        gym.set_articulation_kinematic_states(self.gpu_reset_kinematic_state_command_array)

        # Randomize rigid body material
        total_num_envs = reset_buf.shape[0]
        total_resets = reset_buf.sum().item()
        if total_num_envs > 10 and total_num_envs == total_resets: # only reset friction if all envs are being reset, otherwise it will cause
            static_friction = torch.rand(total_num_envs, device=device) * 0.95 + 0.05
            dynamic_friction = static_friction * 0.75
        else:
            static_friction = 0.1
            dynamic_friction = 0.075

        self.set_static_friction_buf[:] = static_friction
        self.set_dynamic_friction_buf[:] = dynamic_friction
        gym.set_rigid_material_properties(self.gpu_set_friction_cmd)

    def pre_physics_step(
        self,
        gym: v.Gym,
        scaled_act_buf: torch.Tensor,
        max_velocity: torch.Tensor,
    ) -> None:
        """Apply wrist velocity, joint motor commands, and gravity compensation."""
        # Apply wrist velocity commands
        self.set_root_transform_buf[:] = self.get_root_transform_buf
        self.set_root_vel_buf[:] = torch.clamp(scaled_act_buf[:, :6], -max_velocity, max_velocity)
        gym.set_articulation_kinematic_states(self.gpu_set_kinematic_state_command_array)

        # Apply joint motor commands and antagonistic spring
        self.set_motor_cmd_buf[:] = torch.clamp(scaled_act_buf[:, 6:], 0.0, None)
        self.set_motor_cmd_buf[:] += -0.1 * self.get_joint_pos_buf
        gym.set_motor_forces(self.gpu_set_motor_control_command_array)

        # Gravity compensation on base link
        self.set_force_torque_buf[:, :, 2] = 9.81 * self.link_masses
        gym.set_link_external_forces(self.set_force_torque_cmd_arr)
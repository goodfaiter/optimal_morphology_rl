import math
import torch
import numpy as np
from vlearn.spaces import Box
from typing import Tuple, List
import vlearn as v
import random

# from tools.vlearn.train.envs.environment import EnvironmentGpu
import os
import sys

# sys.path.append(os.path.join(os.path.dirname(__file__), '/workspace/tools/vlearn/train/envs/'))
# from environment import EnvironmentGpu
from vlearn_train.envs.environment import EnvironmentGpu

# from vlearn.train.envs.environment import EnvironmentGpu
from vlearn.torch_utils.torch_jit_utils import scale, quat_mul, quat_conjugate, v_rpy_from_quat, quat_rotate


def vec3_to_numpy(vec: v.Vec3) -> np.ndarray:
    return np.array([vec.x, vec.y, vec.z], dtype=np.float32)


def quat_to_numpy(quat: v.Quat) -> np.ndarray:
    return np.array([quat.x, quat.y, quat.z, quat.w], dtype=np.float32)


def numpy_to_vec3(arr: np.ndarray) -> v.Vec3:
    return v.Vec3(arr[0], arr[1], arr[2])


def numpy_to_quat(arr: np.ndarray) -> v.Quat:
    return v.Quat(arr[0], arr[1], arr[2], arr[3])


def world_down_in_robot_frame_from_quat_robot_to_world(quat_robot_to_world: torch.Tensor) -> torch.Tensor:
    quat_world_to_robot = quat_conjugate(quat_robot_to_world)
    down_in_world = (
        torch.tensor([0, 0, -1], device=quat_world_to_robot.device, dtype=quat_world_to_robot.dtype)
        .unsqueeze(0)
        .expand(quat_world_to_robot.shape[0], -1)
    )
    return quat_rotate(quat_world_to_robot, down_in_world)


def palm_down_in_world_frame_from_quat_robot_to_world(quat_robot_to_world: torch.Tensor) -> torch.Tensor:
    palm_down_in_world = (
        torch.tensor([0, -1, 0], device=quat_robot_to_world.device, dtype=quat_robot_to_world.dtype)
        .unsqueeze(0)
        .expand(quat_robot_to_world.shape[0], -1)
    )
    return quat_rotate(quat_robot_to_world, palm_down_in_world)

def rotate_by_quat_A_to_B(quat_A_to_B: torch.Tensor, vec_A: torch.Tensor) -> torch.Tensor:
    return quat_rotate(quat_A_to_B, vec_A)


def pen_forward_in_world_frame_from_quat_pen_to_world(quat_pen_to_world: torch.Tensor) -> torch.Tensor:
    pen_forward_in_world = (
        torch.tensor([1, 0, 0], device=quat_pen_to_world.device, dtype=quat_pen_to_world.dtype)
        .unsqueeze(0)
        .expand(quat_pen_to_world.shape[0], -1)
    )
    return quat_rotate(quat_pen_to_world, pen_forward_in_world)

def random_uniform_quaternion(num_envs, device, dtype):
    q = torch.randn(num_envs, 4, device=device, dtype=dtype)  # 4 independent Gaussians
    return q / torch.norm(q, dim=1, keepdim=True)  # Normalize to unit length


class HandPenEnvironmentGpu(EnvironmentGpu):
    """
    Based on https://github.com/rayangdn/MorphHand environments
    Morphological hand environment with with pen interaction.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        rendering: bool = False,
        enable_scene_query: bool = False,
        max_episode_length: int = 4 * 50,  # 4 seconds at 50Hz control frequency
        gravity: v.Vec3 = v.Vec3(0, 0, -9.81),
        timestep: float = 1/120,  # 120Hz sim frequency
        frame_skip: int = 2,  # 60Hz control step
        spacing: float = 0.5,
        initial_is_paused: bool = False,
        send_interrupt: bool = False,
        print_hash: bool = False,
        max_contact_pairs_per_env: int = 64,
        has_self_collisions: bool = False,
        force_mass_inertia_computation: bool = False,
        with_window: bool = True,
        pen_size: str = "big",
        fixed_hand: bool = False,
    ):

        super().__init__(
            num_envs,
            device,
            rendering,
            enable_scene_query,
            max_episode_length,
            timestep,
            frame_skip,
            spacing,
            gravity,
            initial_is_paused=initial_is_paused,
            send_interrupt=send_interrupt,
            up_axis=v.Vec3(0, 0, 1),
            print_hash=print_hash,
            max_contact_pairs_per_env=max_contact_pairs_per_env,
            with_window=with_window,
        )

        assert pen_size in ["small", "mid", "big"], f"Invalid pen size: {pen_size}"

        self.num_envs_per_set = 32
        if self.num_envs % self.num_envs_per_set != 0:
            raise ValueError(f"num_envs must be a multiple of {self.num_envs_per_set}.")
        self.num_envs = [self.num_envs_per_set] * (self.num_envs // self.num_envs_per_set)
        self.device = device
        self.max_episode_length = max_episode_length
        self.pen_size = pen_size
        self.force_mass_inertia_computation = force_mass_inertia_computation
        self.fixed_hand = fixed_hand

        # Create environments
        self.create_envs()

        # Setup action and observation spaces
        self._setup_action_space()
        self._setup_observation_space()

        # Allocate buffers
        self.allocate_buffers()

        # Create GPU commands
        self.create_gpu_commands()

        # Finalize gym
        self.gym.finalize()
        self.gym.update_scene_dependent_components()

        if self.gym.get_render() is not None:
            #  v.Vec3(-0.132636, 0.030610, 0.021915), v.Vec3(0.991742, 0.127220, -0.016226)
            #  v.Vec3(-0.125011, -0.001044, 0.065207), v.Vec3(0.880567, 0.397776, -0.257634)
            # Camera eye, dir: Vec3(-0.484404, -0.760840, 0.387874), Vec3(0.473463, 0.797717, -0.373469)
            self.gym_render.reset_camera(v.Vec3(-0.484404, -0.760840, 0.387874), v.Vec3(0.473463, 0.797717, -0.373469))

        self.info["rewards"] = {}

    def create_envs(self):
        """Create simulation environments."""
        # Create environment definition
        self.env_def_handle = self.gym.create_environment_def("hand_env")
        env_def = self.gym.get_environment_def(self.env_def_handle)

        # Load appropriate hand model
        hand_file = "/workspace/data/llm/cad_parameters_a_hand_can_pick_up_a_berry.vsim"

        env_def.import_definitions(
            hand_file,
            fixed=self.fixed_hand,
            use_visual_mesh=True,
            merge_fixed_joints=True,
            force_mass_computation=False,
            force_inertia_computation=False,
        )

        # Configure articulation
        self.def_handle = env_def.get_articulation_def_handle(0)
        self.art_def = env_def.get_articulation_def(self.def_handle)
        self.art_def.has_self_collisions = False

        self.art_def.enable_control_type(v.ArticulationControlType.MOTOR, True)

        # Create articulation
        self.arti_handle = env_def.create_articulation(self.def_handle, v.Transform(v.Quat(0, 0, 0, 1), v.Vec3(0, 0, 0)), "hand")

        # Validate dimensions
        self.num_joints = self.art_def.get_num_joint_dof_defs()
        self.num_links = self.art_def.get_num_link_defs()
        self.num_motors = self.art_def.get_num_motor_defs()
        self.link_masses = torch.zeros(self.num_links, dtype=torch.float32, device=self.device)
        for i in range(self.num_links):
            self.link_masses[i] = self.art_def.get_link_def(i).mass

        for i in range(self.num_joints):
            joint_def = self.art_def.get_joint_def(i)
            print(joint_def)

        for i in range(self.num_links):
            link_def = self.art_def.get_link_def(i)
            print(link_def)

        self.thumb_motor_index = -1
        for i in range(self.num_joints):
            motor_def = self.art_def.get_motor_def(i)
            print(i, motor_def)
            if motor_def.name == "thumb_0_proximal_abd_joint":
                self.thumb_motor_index = i
        if self.thumb_motor_index == -1:
            raise ValueError("Could not find thumb motor index")

        for i in range(self.num_links):
            link_def = self.art_def.get_link_def(i)
            print(i, link_def)

        # Rigid Material Frictions
        # Create rigid material
        rigid_mat = v.RigidMaterial()
        rigid_mat.dynamic_friction = 0.01
        rigid_mat.static_friction = 0.01
        rigid_mat.restitution = 0.0
        self.rigid_mat_handle = env_def.create_rigid_material(rigid_mat)
        for i in range(self.art_def.get_num_link_defs()):
            env_def.assign_rigid_material_to_articulation_link(self.def_handle, self.rigid_mat_handle, i)

        # Create table
        self._create_table(env_def)

        # Load and configure pen
        self._create_pen(env_def)

        env_set_offsets = []
        spacing = math.sqrt(self.num_envs_per_set)
        for i in range(len(self.num_envs)):
            # set x and y into square grid:
            grid_size = math.ceil(math.sqrt(len(self.num_envs)))
            x = (i % grid_size) * spacing
            y = (i // grid_size) * spacing
            env_set_offsets.append(v.Vec3(x, y, 0))

        env_def.finalize()
        super().create_envs(self.env_def_handle, env_set_offsets=env_set_offsets)

    def _setup_action_space(self):
        """Configure action space dimensions."""
        # self.num_actions = self.num_joints + 6  # Revolute joints + base link velocity
        self.num_actions = 6 + 1 + 1  # wrist velocities (6) + 1 grasp command + 1 thumb command

        self.action_space = Box(
            low=np.full(self.num_actions, -1.0, dtype=np.float32), high=np.full(self.num_actions, 1.0, dtype=np.float32), dtype=np.float32
        )

        self.velocity_scale = torch.tensor([1.0, 1.0, 1.0, 0.2, 0.2, 0.2], dtype=torch.float32, device=self.device)
        self.revolute_scale = torch.full((1,), 0.25, device=self.device)

    def _setup_observation_space(self):
        """Configure observation space dimensions."""
        # Joint positions + velocities + ball state + actions
        self.num_obs = 3  # hand position in world
        self.num_obs += 4 # hand orientation in world (quat)
        self.num_obs += 3  # hand angular velocity in hand frame
        self.num_obs += 3  # hand linear velocity in hand frame
        self.num_obs += self.num_joints  # joint positions
        self.num_obs += self.num_joints  # joint velocities
        self.num_obs += self.num_actions # current actions
        self.num_obs += self.num_actions  # last actions
        self.num_obs += 3  # pen position in hand frame
        self.num_obs += 4  # pen orientation in hand frame (quat)
        self.num_obs += 3  # pen linear velocity in hand frame
        self.num_obs += 3  # pen angular velocity in hand frame
        self.num_obs += 4  # Pen goal orientation quaternion in world frame
        self.num_obs += 1  # Pen goal height


                # # self.robot_pos_in_world,  # base link position in world
                # world_down_in_robot_frame,  # world down in robot frame (for orientation)
                # self.robot_angular_velocity_in_robot_frame,  # Base link angular velocity in robot frame
                # self.robot_linear_velocity_in_robot_frame,  # Base link linear velocity in robot frame
                # self.get_joint_pos_buf,  # Joint positions
                # self.get_joint_vel_buf,  # Joint velocities
                # self.act_buf,  # Current actions
                # self.last_act_buf,  # Last actions
                # pen_down_in_world,  # pen down direction in world frame
                # self.pen_position_in_robot_frame,  # Pen position
                # self.pen_orientation_in_robot_frame,  # Pen orientation in robot frame (quat)
                # self.pen_linear_velocity_in_robot_frame,  # Pen linear velocity in robot frame
                # self.pen_angular_velocity_in_robot_frame,  # Pen angular velocity in robot frame
                # self.pen_goal_quat_pen_to_world,

        print(f"Observation space size: {self.num_obs}")

        self.observation_space = Box(
            low=np.full(self.num_obs, np.finfo("f").min, dtype=np.float32),
            high=np.full(self.num_obs, np.finfo("f").max, dtype=np.float32),
            dtype=np.float32,
        )

    def _create_table(self, env_def):
        """Create a table for the pen to interact with."""
        rgb_mat = v.RGBMaterial()
        rgb_mat.color = v.Vec3(1, 1, 0)
        rgb_mat.specular = 40
        rgb_mat.spec_intensity = 0.25
        rgb_mat_handle = env_def.create_rgb_material(rgb_mat)

        self.table_half_size = v.Vec3(0.2, 0.3, 0.01)
        table_def_handle = env_def.create_box_def(
            half_size=self.table_half_size, name="table", fixed=True, rgb_material_handle=rgb_mat_handle
        )
        table_handle = env_def.create_rigid_body(
            table_def_handle, v.Transform(v.Quat(0, 0, 0, 1), v.Vec3(0, 0, -self.table_half_size.z)), "table"
        )

        self.table_bounds = torch.tensor(
            [
                [-self.table_half_size.x, self.table_half_size.x],
                [-self.table_half_size.y, self.table_half_size.y],
                [0.0, 0.3],
            ],
            device=self.device,
            dtype=torch.float32,
        )

    def _create_pen(self, env_def):
        """Load and configure the pen object."""
        # rgb_mat = v.RGBMaterial()
        # rgb_mat.color = v.Vec3(0, 1, 0)
        # rgb_mat.specular = 40
        # rgb_mat.spec_intensity = 0.25
        # rgb_mat_handle = env_def.create_rgb_material(rgb_mat)

        # if self.pen_size == "small":
        #     pen_half_size = v.Vec3(0.005, 0.075, 0.005)
        # elif self.pen_size == "mid":
        #     pen_half_size = v.Vec3(0.05, 0.08, 0.05)
        # else:  # big
        #     pen_half_size = v.Vec3(0.07, 0.11, 0.07)

        # pen_def_handle = env_def.create_box_def(half_size=pen_half_size, name="pen", fixed=False, rgb_material_handle=rgb_mat_handle)

        # # Set initial pen pose
        # self._init_pen_transforms = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32)  # quat (x,y,z,w) + pos (x,y,z)
        # pen_root_trans_init = v.Transform(numpy_to_quat(self._init_pen_transforms[:4]), numpy_to_vec3(self._init_pen_transforms[4:]))

        # # Create pen
        # self.pen_handle = env_def.create_rigid_body(pen_def_handle, pen_root_trans_init, "pen")

        """Load and configure the pen ob0, ject."""
        # Load appropriate pen model
        pen_files = {
            "small": "/workspace/assets/objects/pen_small.vsim",
            "mid": "/workspace/assets/objects/pen_mid.vsim",
            "big": "/workspace/assets/objects/pen_big.vsim",
        }
        print(f"Loading pen of size: {self.pen_size}")

        env_def.import_definitions(
            pen_files[self.pen_size],
            fixed=False,
            use_visual_mesh=False,
            force_mass_computation=False,
            force_inertia_computation=False,
        )

        # Get pen definition
        pen_def_handle = env_def.get_rigid_body_def_handle_by_name("pen")

        # # Set initial pen pose
        self._init_pen_transforms = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32)  # quat (x,y,z,w) + pos (x,y,z)
        pen_root_trans_init = v.Transform(numpy_to_quat(self._init_pen_transforms[:4]), numpy_to_vec3(self._init_pen_transforms[4:]))

        # Create pen
        self.pen_handle = env_def.create_rigid_body(pen_def_handle, pen_root_trans_init, "pen")

    def allocate_buffers(self):
        """Allocate GPU buffers for state and control."""
        super().allocate_buffers()

        ### Env Buffers
        self.inverse_reset_buf = torch.zeros(self.total_num_envs, device=self.device, dtype=torch.bool)

        ### Hand Buffers
        # Reset buffers
        self.reset_joint_pos_buf = torch.zeros((self.total_num_envs, self.num_joints), device=self.device, dtype=torch.float32)
        self.reset_joint_vel_buf = torch.zeros((self.total_num_envs, self.num_joints), device=self.device, dtype=torch.float32)
        self.reset_root_transform_buf = torch.zeros((self.total_num_envs, 7), device=self.device, dtype=torch.float32)  # quat (4) + pos (3)
        self.reset_root_vel_buf = torch.zeros((self.total_num_envs, 6), device=self.device, dtype=torch.float32)

        # Set joint force controls
        self.set_motor_cmd_buf = torch.zeros((self.total_num_envs, self.num_joints), device=self.device, dtype=torch.float32)

        # Get joint states
        self.get_joint_pos_buf = torch.zeros((self.total_num_envs, self.num_joints), device=self.device, dtype=torch.float32)
        self.get_joint_vel_buf = torch.zeros((self.total_num_envs, self.num_joints), device=self.device, dtype=torch.float32)
        self.get_root_transform_buf = torch.zeros((self.total_num_envs, 7), device=self.device, dtype=torch.float32)  # quat (4) + pos (3)
        self.get_root_vel_buf = torch.zeros((self.total_num_envs, 6), device=self.device, dtype=torch.float32)

        # Robot states
        self.robot_pos_in_world = torch.zeros((self.total_num_envs, 3), device=self.device, dtype=torch.float32)
        self.robot_quat_robot_to_world = torch.zeros((self.total_num_envs, 4), device=self.device, dtype=torch.float32)
        self.robot_linear_velocity_in_world = torch.zeros((self.total_num_envs, 3), device=self.device, dtype=torch.float32)
        self.robot_angular_velocity_in_world = torch.zeros((self.total_num_envs, 3), device=self.device, dtype=torch.float32)

        # Pen state buffers
        self.get_pen_pos_buf = torch.zeros((self.total_num_envs, 7), device=self.device, dtype=torch.float32)
        self.get_pen_vel_buf = torch.zeros((self.total_num_envs, 6), device=self.device, dtype=torch.float32)

        # Pen kinematics control buffers
        self.set_pen_pos_buf = torch.zeros((self.total_num_envs, 7), device=self.device, dtype=torch.float32)
        self.set_pen_vel_buf = torch.zeros((self.total_num_envs, 6), device=self.device, dtype=torch.float32)

        # Pen initial state
        self.gpu_init_pen_transforms = torch.tensor(self._init_pen_transforms, dtype=torch.float32, device=self.device)
        self.gpu_init_pen_velocities = torch.zeros(6, dtype=torch.float32, device=self.device)

        self.set_pen_pos_buf[:] = self.gpu_init_pen_transforms
        self.set_pen_vel_buf[:] = self.gpu_init_pen_velocities

        # Goal pen rotation (upright)
        self.pen_goal_quat_pen_to_world = (
            torch.tensor([0, 0, 0, 1], device=self.device, dtype=torch.float32).unsqueeze(0).repeat(self.total_num_envs, 1)
        )

        # Last action buffer
        self.last_act_buf = torch.zeros_like(self.act_buf)
        self.scaled_act_buf = torch.zeros_like(self.act_buf)

        # Gravity Comp Buffers
        self.set_joint_pos_buf = torch.zeros((self.total_num_envs, 0), device=self.device, dtype=torch.float32)
        self.set_joint_vel_buf = torch.zeros((self.total_num_envs, 0), device=self.device, dtype=torch.float32)
        self.set_root_transform_buf = torch.zeros((self.total_num_envs, 7), device=self.device, dtype=torch.float32)  # quat (4) + pos (3)
        self.set_root_vel_buf = torch.zeros((self.total_num_envs, 6), device=self.device, dtype=torch.float32)
        self.set_force_torque_buf = torch.zeros((self.total_num_envs, self.num_links, 6), dtype=torch.float32, device=self.device)

        # Rigid Material Buffers
        self.set_static_friction_buf = torch.zeros(len(self.num_envs), dtype=torch.float32, device=self.device)
        self.set_dynamic_friction_buf = torch.zeros(len(self.num_envs), dtype=torch.float32, device=self.device)

        self.pen_goal_z = torch.ones((self.total_num_envs, 1), device=self.device, dtype=torch.float32) * (self.table_bounds[2, 1] - 0.1)

    def create_gpu_commands(self):
        """Create GPU command arrays for efficient state queries and control."""
        ### Hand Commands
        # Kinematic state command
        reset_kin_cmd = self.env_group.create_articulation_kinematic_state_command(
            v.wrap_gpu_buffer(self.reset_joint_pos_buf),
            v.wrap_gpu_buffer(self.reset_joint_vel_buf),
            v.wrap_gpu_buffer(self.reset_root_transform_buf),
            v.wrap_gpu_buffer(self.reset_root_vel_buf),
            self.arti_handle,
            (0, self.num_joints),
            (0, 1),
            masks_buffer=v.wrap_gpu_buffer(self.reset_buf),
        )
        self.gpu_reset_kinematic_state_command_array = self.gym.create_gpu_array([reset_kin_cmd])

        set_kin_cmd = self.env_group.create_articulation_kinematic_state_command(
            v.wrap_gpu_buffer(self.set_joint_pos_buf),
            v.wrap_gpu_buffer(self.set_joint_vel_buf),
            v.wrap_gpu_buffer(self.set_root_transform_buf),
            v.wrap_gpu_buffer(self.set_root_vel_buf),
            self.arti_handle,
            (0, 0),
            (0, 1),
            masks_buffer=v.wrap_gpu_buffer(self.inverse_reset_buf),
        )
        self.gpu_set_kinematic_state_command_array = self.gym.create_gpu_array([set_kin_cmd])

        get_kin_cmd = self.env_group.create_articulation_kinematic_state_command(
            v.wrap_gpu_buffer(self.get_joint_pos_buf),
            v.wrap_gpu_buffer(self.get_joint_vel_buf),
            v.wrap_gpu_buffer(self.get_root_transform_buf),
            v.wrap_gpu_buffer(self.get_root_vel_buf),
            self.arti_handle,
            (0, self.num_joints),
            (0, 1),
        )
        self.gpu_get_kinematic_state_command_array = self.gym.create_gpu_array([get_kin_cmd])

        # Motor Command
        set_motor_cmd = self.env_group.create_motor_control_command(
            v.wrap_gpu_buffer(self.set_motor_cmd_buf), self.arti_handle, index_range=[0, self.num_motors]
        )
        self.gpu_set_motor_control_command_array = self.gym.create_gpu_array([set_motor_cmd])

        ### Pen commands
        # Get pen state
        get_pen_kin_cmd = self.env_group.create_rigid_body_kinematic_state_command(
            v.wrap_gpu_buffer(self.get_pen_pos_buf), v.wrap_gpu_buffer(self.get_pen_vel_buf), self.pen_handle
        )
        self.gpu_get_pen_kin_cmd_array = self.gym.create_gpu_array([get_pen_kin_cmd])

        # Set pen state (for reset)
        set_pen_kin_cmd = self.env_group.create_rigid_body_kinematic_state_command(
            v.wrap_gpu_buffer(self.set_pen_pos_buf),
            v.wrap_gpu_buffer(self.set_pen_vel_buf),
            self.pen_handle,
            masks_buffer=v.wrap_gpu_buffer(self.reset_buf),
        )
        self.gpu_set_pen_kin_cmd_array = self.gym.create_gpu_array([set_pen_kin_cmd])

        ### Gravity Comp Commands
        # Create external force command
        set_force_torque_cmd = self.env_group.create_link_external_force_command(
            v.wrap_gpu_buffer(self.set_force_torque_buf), self.arti_handle, [0, self.num_links], force_type=v.ForceType.FORCE_TORQUE
        )

        self.set_force_torque_cmd_arr = self.gym.create_gpu_array([set_force_torque_cmd])

        ### Rigid Material Commands
        set_static_friction_cmd = self.env_group.create_rigid_material_property_command(
            v.RigidMaterialProperty.STATIC_FRICTION,
            v.wrap_gpu_buffer(self.set_static_friction_buf),
            self.rigid_mat_handle,
            v.wrap_gpu_buffer(self.reset_buf),
        )
        set_dynamic_friction_cmd = self.env_group.create_rigid_material_property_command(
            v.RigidMaterialProperty.DYNAMIC_FRICTION,
            v.wrap_gpu_buffer(self.set_dynamic_friction_buf),
            self.rigid_mat_handle,
            v.wrap_gpu_buffer(self.reset_buf),
        )
        self.gpu_set_friction_cmd = self.gym.create_gpu_array([set_static_friction_cmd, set_dynamic_friction_cmd])


    def reset_idx(self):
        self.act_buf[self.reset_buf, :] = 0.0
        self.last_act_buf[self.reset_buf, :] = 0.0

        # Reset Hand Kinematics
        self.reset_joint_pos_buf[self.reset_buf, :] = (
            torch.rand((self.reset_buf.sum(), self.num_joints), device=self.device) * 0.5 * torch.pi
        )
        self.reset_joint_vel_buf[self.reset_buf, :] = 0.0
        self.reset_root_transform_buf[self.reset_buf, :4] = random_uniform_quaternion(self.reset_buf.sum(), device=self.device, dtype=torch.float32)
        self.reset_root_transform_buf[self.reset_buf, 4:] = (
            self.table_bounds[:, 0]
            + 0.1
            + torch.rand((self.reset_buf.sum(), 3), device=self.device) * (self.table_bounds[:, 1] - self.table_bounds[:, 0] - 0.1)
        )
        self.reset_root_vel_buf[self.reset_buf, :] = 0.0
        self.gym.set_articulation_kinematic_states(self.gpu_reset_kinematic_state_command_array)

        # Reset pen state with noise
        self.set_pen_pos_buf[self.reset_buf, :4] = random_uniform_quaternion(self.reset_buf.sum(), device=self.device, dtype=torch.float32)
        self.set_pen_pos_buf[self.reset_buf, 4:] = (
            self.table_bounds[:, 0]
            + 0.1
            + torch.rand((self.reset_buf.sum(), 3), device=self.device) * (self.table_bounds[:, 1] - self.table_bounds[:, 0] - 0.1)
        )
        self.set_pen_vel_buf[self.reset_buf, :] = 0.0
        self.gym.set_rigid_body_kinematic_states(self.gpu_set_pen_kin_cmd_array)

        # Random height
        self.pen_goal_z[self.reset_buf] = (self.table_bounds[2, 0] + 0.05 + torch.rand(self.reset_buf.sum(), device=self.device) * (self.table_bounds[2, 1] - self.table_bounds[2, 0] - 0.05)).unsqueeze(-1)

        # Random orientation vector
        self.pen_goal_quat_pen_to_world[self.reset_buf, :] = random_uniform_quaternion(self.reset_buf.sum(), device=self.device, dtype=torch.float32)

        # Reset progress
        self.progress_buf[self.reset_buf] = 0

        # Randomize rigid body material
        # Write to buffers
        friction = torch.rand(len(self.num_envs), device=self.device) * 0.7 + 0.3  # [0.3, 1.0]
        self.set_static_friction_buf[:] = friction
        self.set_dynamic_friction_buf[:] = friction
        self.gym.set_rigid_material_properties(self.gpu_set_friction_cmd)

    def reset(self):
        obs, _ = super().reset()
        self.refresh_buffers()
        return obs, {}

    def pre_physics_step(self, actions: torch.Tensor):
        self.last_act_buf[:] = self.act_buf[:]
        self.act_buf[:] = actions
        self.scaled_act_buf[:, :6] = scale(self.act_buf[:, :6], -self.velocity_scale, self.velocity_scale)
        self.scaled_act_buf[:, 6:] = scale(self.act_buf[:, 6:], -self.revolute_scale, self.revolute_scale)

        # Apply wrist velocity commands
        self.set_root_transform_buf[:] = self.get_root_transform_buf
        self.set_root_vel_buf[:] = self.scaled_act_buf[:, :6]  # note that vel is first 3 angular, last 3 linear
        self.gym.set_articulation_kinematic_states(self.gpu_set_kinematic_state_command_array)

        # Apply joint motor commands
        self.set_motor_cmd_buf[:] = self.scaled_act_buf[:, 6].unsqueeze(-1).expand(-1, self.num_motors)
        self.set_motor_cmd_buf[:, self.thumb_motor_index] = self.scaled_act_buf[:, 7]  # thumb command
        self.gym.set_motor_forces(self.gpu_set_motor_control_command_array)

        # Gravity compensation on base link
        self.set_force_torque_buf[:, :, 2] = 9.81 * self.link_masses
        self.gym.set_link_external_forces(self.set_force_torque_cmd_arr)

    def refresh_buffers(self):
        """Refresh all state buffers from simulation."""
        self.gym.get_articulation_kinematic_states(self.gpu_get_kinematic_state_command_array)
        self.gym.get_rigid_body_kinematic_states(self.gpu_get_pen_kin_cmd_array)

    def post_physics_step(self):
        """Post-step processing."""
        self.progress_buf[:] += 1

        # Check for episode termination
        self.reset_buf[:] = torch.logical_or(self.term_buf, self.trunc_buf)
        self.inverse_reset_buf[:] = ~self.reset_buf
        self.reset_idx()

        # Update state
        self.refresh_buffers()
        self.compute_observations()
        self.compute_reward_termination_truncation()

    def compute_observations(self):
        """Construct observation vector."""
        self.robot_pos_in_world = self.get_root_transform_buf[:, 4:7]
        self.robot_quat_robot_to_world = self.get_root_transform_buf[:, 0:4]
        self.robot_linear_velocity_in_world = self.get_root_vel_buf[:, 3:6]
        self.robot_angular_velocity_in_world = self.get_root_vel_buf[:, :3]

        # self.robot_angular_velocity_in_robot_frame = rotate_by_quat_A_to_B(quat_conjugate(self.robot_quat_robot_to_world), self.robot_angular_velocity_in_world)
        # self.robot_linear_velocity_in_robot_frame = rotate_by_quat_A_to_B(quat_conjugate(self.robot_quat_robot_to_world), self.robot_linear_velocity_in_world)

        self.pen_position_in_world = self.get_pen_pos_buf[:, 4:7]
        self.pen_quat_pen_to_world = self.get_pen_pos_buf[:, 0:4]
        self.pen_linear_velocity_in_world = self.get_pen_vel_buf[:, 3:6]
        self.pen_angular_velocity_in_world = self.get_pen_vel_buf[:, :3]

        self.pen_position_in_robot_frame = rotate_by_quat_A_to_B(quat_conjugate(self.robot_quat_robot_to_world), self.get_pen_pos_buf[:, 4:7] - self.robot_pos_in_world)
        self.pen_orientation_in_robot_frame = quat_mul(quat_conjugate(self.robot_quat_robot_to_world), self.get_pen_pos_buf[:, 0:4])
        self.pen_linear_velocity_in_robot_frame = rotate_by_quat_A_to_B(quat_conjugate(self.robot_quat_robot_to_world), self.get_pen_vel_buf[:, 3:6])
        self.pen_angular_velocity_in_robot_frame = rotate_by_quat_A_to_B(quat_conjugate(self.robot_quat_robot_to_world), self.get_pen_vel_buf[:, :3])

        self.obs_buf[:] = torch.cat(
            [
                self.robot_pos_in_world,  # robot_position in world frame
                self.robot_quat_robot_to_world,  # robot orientation in world frame (quat)
                self.robot_linear_velocity_in_world,  # Base link linear velocity in world frame
                self.robot_angular_velocity_in_world,  # Base link angular velocity in world frame
                self.get_joint_pos_buf,  # Joint positions
                self.get_joint_vel_buf,  # Joint velocities
                self.act_buf,  # Current actions
                self.last_act_buf,  # Last actions
                self.pen_position_in_world,  # pen position in world frame
                self.pen_quat_pen_to_world,  # Pen orientation in robot frame (quat)
                self.pen_linear_velocity_in_world,  # Pen linear velocity in robot frame
                self.pen_angular_velocity_in_world,  # Pen angular velocity in robot frame
                self.pen_goal_quat_pen_to_world,
                self.pen_goal_z,
            ],
            dim=-1,
        )

    def compute_reward_termination_truncation(self):
        self.rew_buf[:] = 0.0

        # Pen Height
        pen_height_error = self.pen_goal_z.squeeze(-1) - self.pen_position_in_world[:, 2]
        pen_height_error_normalized = pen_height_error / 0.1
        height_reward = torch.exp(-2.0 * pen_height_error_normalized**2)
        self.info["rewards"]["height_reward"] = height_reward.sum().item() / self.total_num_envs
        self.rew_buf[:] += 2.0 * height_reward

        # Reward pen upright orientation
        quat_diff = quat_mul(quat_conjugate(self.pen_quat_pen_to_world), self.pen_goal_quat_pen_to_world)
        pen_alignment = torch.abs(quat_diff[:, 3])  # alignment is the absolute value of the w component of the quat diff
        pen_alignment_clipped = torch.clamp(pen_alignment, max=1.0)  # don't reward perfect alignment to allow for exploration
        pen_alignment_reward = torch.exp(-2.0 * (1.0 - pen_alignment_clipped) ** 2)
        self.info["rewards"]["rot_reward"] = pen_alignment_reward.sum().item() / self.total_num_envs
        self.rew_buf[:] += 2.0 * pen_alignment_reward

        # Reward distance between pen and hand
        dist = torch.norm(self.robot_pos_in_world - self.pen_position_in_world, dim=-1)
        dist_clipped = torch.clamp(dist, min=0.0)  # don't reward getting too close to allow for exploration
        dist_clipped_normalized = dist_clipped / 0.2
        dist_rew = torch.exp(-1.0 * dist_clipped_normalized**2)
        self.info["rewards"]["dist_reward"] = dist_rew.sum().item() / self.total_num_envs
        self.rew_buf[:] += 1.0 * dist_rew

        # Reward robot alignment of palm with direction to the pen
        hand_to_pen_dir_in_world = torch.nn.functional.normalize(self.robot_pos_in_world - self.pen_position_in_world, dim=-1)
        palm_down_in_world = palm_down_in_world_frame_from_quat_robot_to_world(self.get_root_transform_buf[:, 0:4])
        alignment = torch.sum(hand_to_pen_dir_in_world * palm_down_in_world, dim=-1)
        alignment_clipped = torch.clamp(alignment, max=0.75)  # don't reward perfect alignment to allow for exploration
        alignment_rew = torch.exp(-2.0 * (1 - alignment_clipped) ** 2)
        self.info["rewards"]["alignment_reward"] = alignment_rew.sum().item() / self.total_num_envs
        self.rew_buf[:] += 1.0 * alignment_rew

        # Penalize large actions to encourage smooth control
        action_penalty = torch.sum(torch.abs(self.act_buf), dim=-1)
        action_penalty_normalized = action_penalty / self.num_actions
        action_penalty_reward = torch.exp(-1.0 * action_penalty_normalized**2)
        self.info["rewards"]["action_penalty"] = action_penalty_reward.sum().item() / self.total_num_envs
        self.rew_buf[:] += 0.01 * action_penalty_reward

        # Action smoothness
        action_smoothness_penalty = torch.sum(torch.abs(self.act_buf - self.last_act_buf), dim=-1)
        action_smoothness_penalty_normalized = action_smoothness_penalty / self.num_actions
        action_smoothness_reward = torch.exp(-1.0 * action_smoothness_penalty_normalized**2)
        self.info["rewards"]["action_smoothness_penalty"] = action_smoothness_reward.sum().item() / self.total_num_envs
        self.rew_buf[:] += 0.001 * action_smoothness_reward

        # Penalize object drops and end episode
        drop_penalty = self.pen_position_in_world[:, 2] < -0.1
        drop_reward = -1.0 * drop_penalty
        self.info["rewards"]["drop_penalty"] = drop_reward.sum().item() / self.total_num_envs
        self.rew_buf[:] += 1.0 * drop_reward

        # Penalize hand out of table bounds
        out_of_bounds = torch.logical_or(self.robot_pos_in_world < self.table_bounds[:, 0], self.robot_pos_in_world > self.table_bounds[:, 1])
        bounds_penalty = torch.any(out_of_bounds, dim=-1)
        bounds_reward = -1.0 * bounds_penalty
        self.info["rewards"]["bounds_penalty"] = bounds_reward.sum().item() / self.total_num_envs
        self.rew_buf[:] += 1.0 * bounds_reward

        # Terminations
        self.term_buf[:] = torch.logical_or(drop_penalty, bounds_penalty)

        # Truncations
        self.trunc_buf[:] = self.progress_buf >= self.max_episode_length


if __name__ == "__main__":
    from vlearn.utils import get_VL_VISUAL_TESTS

    # Configuration
    config = {
        "num_envs": 1,
        "rendering": True,
        "with_window": get_VL_VISUAL_TESTS(),
        "max_episode_length": 10000000,
        "pen_size": "big",  # 'small', 'mid', or 'big'
        "fixed_hand": False,
    }

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda:0")

    # Create environment
    envs = HandPenEnvironmentGpu(device=device, **config)
    obs, _ = envs.reset()

    gym = v.get_gym()
    render = gym.get_render()

    # Setup interactive control
    if render is not None:
        # render.reset_camera(v.Vec3(0.0, 0.2, 0.2), v.Vec3(0.023272, -0.929027, -0.369280))
        render.capped_step = True
        render.set_paused(False)

        # UI elements
        reset_box = v.UserCheckbox("Reset", False)
        render.register_menu_item(reset_box)

        # Create sliders based on control mode
        sliders = []

        for i in range(envs.num_actions):
            if i < 3:
                name = f"Angular_Vel_{i}"
            elif i < 6:
                name = f"Linear_Vel_{i - 3}"
            else:
                name = f"DOF_{i - 6}"
            sliders.append(v.UserSlider(name, -1, 1, 0.0))
            render.register_menu_item(sliders[-1])

        def control_by_menu():
            """Read control inputs from UI."""
            if reset_box.get_value():
                envs.reset()
                reset_box.set_value(False)

            actions = torch.tensor([slider.get_value() for slider in sliders], dtype=torch.float32, device=device)

            return actions.unsqueeze(0).expand(config["num_envs"], -1)

        control_fn = control_by_menu

    # Main simulation loop
    print("\nStarting simulation...")
    step = 0
    while not envs.render_finished:
        actions = control_fn()
        obs, rewards, terminations, truncations, infos = envs.step(actions)
        step += 1

    print(f"Simulation completed after {step} steps")

    """Simple test script to create a hand with zero gravity and move via sliders."""

    # Configuration

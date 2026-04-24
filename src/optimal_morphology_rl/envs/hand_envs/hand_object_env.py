import math
import torch
import numpy as np
from vlearn.spaces import Box
from typing import Dict, Tuple, List
import vlearn as v
import random

# from tools.vlearn.train.envs.environment import EnvironmentGpu
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "/workspace/tools/vlearn/train/envs/"))
from environment import EnvironmentGpu
from optimal_morphology_rl.helpers.buffer import TimeSeriesBuffer
from optimal_morphology_rl.envs.hand_envs.helpers.hand_pen_helpers import (
    world_down_in_robot_frame_from_quat_robot_to_world,
    palm_down_in_world_frame_from_quat_robot_to_world,
    rotate_by_quat_A_to_B,
    pen_forward_in_world_frame_from_quat_pen_to_world,
)
from optimal_morphology_rl.envs.hand_envs.helpers.numpy_vlearn import (
    vec3_to_numpy,
    quat_to_numpy,
    numpy_to_vec3,
    numpy_to_quat,
    random_uniform_quaternion,
    quaternion_to_6d,
)

from vlearn.torch_utils.torch_jit_utils import scale, quat_mul, quat_conjugate, quat_diff_rad


class HandObjectEnvironmentGpu(EnvironmentGpu):
    """
    Based on https://github.com/rayangdn/MorphHand environments
    Morphological hand environment with generic object interaction.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        rendering: bool = False,
        enable_scene_query: bool = True,
        max_episode_length: int = 2 * 60,  # 2 seconds at 60Hz control frequency
        gravity: v.Vec3 = v.Vec3(0, 0, -9.81),
        timestep: float = 1 / 120,  # 120Hz sim frequency
        frame_skip: int = 2,  # 60Hz control step
        spacing: float = 0.5,
        initial_is_paused: bool = False,
        send_interrupt: bool = False,
        print_hash: bool = False,
        force_mass_inertia_computation: bool = False,
        with_window: bool = True,
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
            update_scene_dependent_components_in_step=True,
            enable_deformable_simulation=True,
            compute_deformable_kinematic=True,
            max_deform_deform_points_per_env=64,
            max_deform_rigid_pairs_per_env=128,
            max_deform_rigid_patches_per_env=128,
            max_deform_rigid_points_per_env=2048,
            initial_render_substep=False,
            send_interrupt=send_interrupt,
            up_axis=v.Vec3(0, 0, 1),
            print_hash=print_hash,
            with_window=with_window,
        )

        self.num_envs_per_set = 1
        if self.num_envs % self.num_envs_per_set != 0:
            raise ValueError(f"num_envs must be a multiple of {self.num_envs_per_set}.")
        self.num_envs = [self.num_envs_per_set] * (self.num_envs // self.num_envs_per_set)
        self.device = device
        self.max_episode_length = max_episode_length
        self.reward_object: str = "mug"
        self.force_mass_inertia_computation = force_mass_inertia_computation
        self.fixed_hand = fixed_hand
        self.max_contact_pairs_per_env = 64
        self.num_hist = 3
        self.hist_stride = 10
        self.obs_history_length = 1 + (self.num_hist - 1) * self.hist_stride
        self._cube = None

        # Generic object map for loading and canonical initial transforms.
        # self.object_creation_order = ["pen", "tomato", "knife", "mug"]
        self.object_creation_order = [self.reward_object]
        self.object_asset_map: Dict[str, (str, bool)] = {
            "pen": ("/workspace/assets/objects/pen_big.vsim", False),
            "tomato": ("/workspace/assets/objects/tomato.vsim", False),
            "tomato_deformable": ("/workspace/assets/objects/tomato_deformable.vsim", True),
            "knife": ("/workspace/assets/objects/kitchen_knife.vsim", False),
            "mug": ("/workspace/assets/objects/mug.vsim", False),
            "ModelRoot": ("/workspace/tools/vlearn/assets/vsim/TEXTURE_BUNNY/textured_bunny.vsim", True),
        }
        self.object_visual_mesh_map: Dict[str, bool] = {
            "pen": False,
            "tomato": False,
            "tomato_deformable": True,
            "knife": True,
            "mug": True,
            "ModelRoot": False,
        }
        self.object_init_transform_map: Dict[str, np.ndarray] = {
            "pen": np.array([0, 0, 0, 1, 0.0, 0.0, 0.1], dtype=np.float32),
            "tomato": np.array([0, 0, 0, 1, 0.0, 0.1, 0.1], dtype=np.float32),
            "tomato_deformable": np.array([0, 0, 0, 1, 0.0, 0.1, 0.1], dtype=np.float32),
            "knife": np.array([0, 0, 0, 1, 0.0, 0.0, 0.1], dtype=np.float32),
            "tomato": np.array([0, 0, 0, 1, 0.0, 0.1, 0.1], dtype=np.float32),
            "mug": np.array([0, 0, 0, 1, 0.0, 0.2, 0.1], dtype=np.float32),
            "ModelRoot": np.array([0, 0, 0, 1, 0.0, 0.0, 0.0], dtype=np.float32),
        }

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

        # Cache transform indices needed to identify pen-hand contacts per environment.
        self._cache_contact_transform_indices()

        if self.gym.get_render() is not None:
            #  v.Vec3(-0.132636, 0.030610, 0.021915), v.Vec3(0.991742, 0.127220, -0.016226)
            #  v.Vec3(-0.125011, -0.001044, 0.065207), v.Vec3(0.880567, 0.397776, -0.257634)
            # Camera eye, dir: Vec3(-0.484404, -0.760840, 0.387874), Vec3(0.473463, 0.797717, -0.373469)
            self.gym_render.reset_camera(v.Vec3(-0.484404, -0.760840, 0.387874), v.Vec3(0.473463, 0.797717, -0.373469))
            # Camera eye, dir: Vec3(-0.156717, -0.131831, 0.434795), Vec3(0.425524, 0.531329, -0.732543)
            # self.gym_render.reset_camera(v.Vec3(-0.156717, -0.131831, 0.434795), v.Vec3(0.425524, 0.531329, -0.732543)) # for hand variants

        self.info["rewards"] = {}

    def create_envs(self):
        """Create simulation environments."""
        # Create environment definition
        self.env_def_handle = self.gym.create_environment_def("hand_env")
        env_def = self.gym.get_environment_def(self.env_def_handle)

        # Load appropriate hand model
        # hand_file = "/workspace/data/llm/cad_parameters_a_hand_can_pick_up_a_berry.vsim"
        # hand_file = "/workspace/data/llm/cad_parameters_a_hand_can_pick_up_a_berry_copy.vsim"
        # hand_file = "/workspace/data/llm/cad_parameters_a_hand_can_pick_up_a_berry_converted.vsim"

        # hand_file = "/workspace/data/llm/variant_0_urdf_params/variant_0_urdf_params.vsim"
        # hand_file = "/workspace/data/llm/variant_1_urdf_params/variant_1_urdf_params.vsim"
        # hand_file = "/workspace/data/llm/variant_2_urdf_params/variant_2_urdf_params.vsim"

        # hand_file = "/workspace/data/llm/iteration_02/variant_2_upgrade_0_hand.vsim"
        # hand_file = "/workspace/data/llm/iteration_02/variant_2_upgrade_1_hand.vsim"
        hand_file = "/workspace/data/llm/iteration_02/variant_2_upgrade_2_hand.vsim"

        env_def.import_definitions(
            hand_file,
            fixed=self.fixed_hand,
            use_visual_mesh=False,
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
            # print(joint_def)

        for i in range(self.num_links):
            link_def = self.art_def.get_link_def(i)
            # print(link_def)

        self.thumb_motor_index = -1
        for i in range(self.num_joints):
            motor_def = self.art_def.get_motor_def(i)
            # print(i, motor_def)
            if motor_def.name == "thumb_0_proximal_abd_joint":
                self.thumb_motor_index = i

        for i in range(self.num_links):
            link_def = self.art_def.get_link_def(i)
            # print(i, link_def)

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

        # Load all supported objects through a generic loader.
        self._create_objects(env_def)

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

    def _cache_contact_transform_indices(self):
        max_envs_in_set = max(self.num_envs)
        self.contact_env_lookup = torch.full((len(self.num_envs), max_envs_in_set), -1, dtype=torch.long, device=self.device)
        self.reward_object_transform_index_by_env = torch.full((self.total_num_envs,), -1, dtype=torch.long, device=self.device)
        self.table_transform_index_by_env = torch.full((self.total_num_envs,), -1, dtype=torch.long, device=self.device)
        self.hand_transform_indices_by_env = torch.full((self.total_num_envs, self.num_links), -1, dtype=torch.long, device=self.device)

        self.hand_transform_indices_by_env[:, :] = torch.tensor(
            [i for i in range(self.num_links)], dtype=torch.long, device=self.device
        ).repeat(self.total_num_envs, 1)
        self.table_transform_index_by_env[:] = self.num_links
        # Transform index layout is hand links, table, then objects in object_creation_order.
        reward_object_offset = self.object_creation_order.index(self.reward_object)
        self.reward_object_transform_index_by_env[:] = self.num_links + 1 + reward_object_offset

        env_flat_index = 0
        for set_index, env_set in enumerate(self.env_sets):
            num_envs_in_set = env_set.get_num_environments()
            for env_index in range(num_envs_in_set):
                self.contact_env_lookup[set_index, env_index] = env_flat_index
                env_flat_index += 1

    def _setup_action_space(self):
        """Configure action space dimensions."""
        # self.num_actions = self.num_joints + 6  # Revolute joints + base link velocity
        self.num_actions = 6 + 1 + 1  # wrist velocities (6) + 1 grasp command + 1 thumb command
        # self.num_actions = 6 + self.num_motors  # wrist velocities (6) + motor commands

        self.action_space = Box(
            low=np.full(self.num_actions, -1.0, dtype=np.float32), high=np.full(self.num_actions, 1.0, dtype=np.float32), dtype=np.float32
        )

        self.velocity_scale = torch.tensor([1.0, 1.0, 1.0, 0.2, 0.2, 0.2], dtype=torch.float32, device=self.device)
        self.force_scale = torch.tensor([0.5, 0.5, 0.5, 1.0, 1.0, 1.0], dtype=torch.float32, device=self.device)
        self.revolute_scale = torch.full((2,), 0.25, device=self.device)
        # self.revolute_scale = torch.full((self.num_motors, ), 0.25, device=self.device)

    def _setup_observation_space(self):
        """Configure observation space dimensions."""
        self.base_num_obs = 3  # hand position in world
        self.base_num_obs += 3  # hand palm down vector orientation in world (3d)
        self.base_num_obs += 3  # hand angular velocity in hand frame
        self.base_num_obs += 3  # hand linear velocity in hand frame
        # self.base_num_obs += self.num_joints  # joint positions
        self.base_num_obs += 2  # average finger curl and thumb abd/add
        # self.base_num_obs += self.num_joints  # joint velocities
        self.base_num_obs += 2  # average finger speed and thumb speed
        self.base_num_obs += self.num_actions  # current actions
        self.base_num_obs += self.num_actions  # last actions
        self.base_num_obs += 3  # reward object position in world frame
        self.base_num_obs += 3  # reward object orientation in world frame (3d)
        self.base_num_obs += 3  # reward object linear velocity in world frame
        self.base_num_obs += 3  # reward object angular velocity in world frame
        self.base_num_obs += 3  # reward object goal orientation in world frame (3d)
        self.base_num_obs += 3  # reward object goal position in world frame
        self.base_num_obs += 3  # reward object goal orientation in hand frame (3d)
        self.base_num_obs += 3  # reward object orientation in hand frame (3d)

        # Joint positions + velocities + ball state + actions
        self.num_obs = self.base_num_obs * self.num_hist

        # Observation and contact history buffer.
        self.obs_history = TimeSeriesBuffer(
            num_envs=self.total_num_envs,
            dim=self.base_num_obs,
            max_size=self.obs_history_length,
            stride=self.hist_stride,
            device=self.device,
        )

        print(f"Observation space size: {self.num_obs} " f"(base={self.base_num_obs}, num_hist={self.num_hist}, stride={self.hist_stride})")

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
        self.table_handle = env_def.create_rigid_body(
            table_def_handle, v.Transform(v.Quat(0, 0, 0, 1), v.Vec3(0, 0, -self.table_half_size.z)), "table"
        )

        self.table_bounds = torch.tensor(
            [
                [-self.table_half_size.x, self.table_half_size.x],
                [-self.table_half_size.y, self.table_half_size.y],
                [0.0, 0.35],
            ],
            device=self.device,
            dtype=torch.float32,
        )

    def _create_objects(self, env_def):
        """Load all supported objects with one generic loader."""
        self.object_handles: Dict[str, int] = {}
        self.object_is_deformable: Dict[str, bool] = {}
        self._init_object_transforms: Dict[str, np.ndarray] = {}

        for object_name in self.object_creation_order:
            is_deformable = self.object_asset_map[object_name][1]
            self.object_is_deformable[object_name] = is_deformable
            self._create_object(
                env_def=env_def,
                object_name=object_name,
                asset_file=self.object_asset_map[object_name][0],
                deforable=is_deformable,
                init_transform=self.object_init_transform_map[object_name],
                use_visual_mesh=self.object_visual_mesh_map[object_name],
            )

        self.reward_object_handle = self.object_handles[self.reward_object]
        self.reward_object_is_deformable = self.object_is_deformable[self.reward_object]
        self._init_reward_object_transforms = self._init_object_transforms[self.reward_object]

    def _create_object(
        self, env_def, object_name: str, asset_file: str, deforable: bool, init_transform: np.ndarray, use_visual_mesh: bool
    ):
        env_def.import_definitions(
            asset_file,
            fixed=False,
            use_visual_mesh=use_visual_mesh,
            force_mass_computation=True if deforable else False,
            force_inertia_computation=False,
        )

        object_root_trans_init = v.Transform(numpy_to_quat(init_transform[:4]), numpy_to_vec3(init_transform[4:]))

        if deforable:
            object_def_handle = env_def.get_deformable_def_handle_by_name(object_name)
            object_handle = env_def.create_deformable(object_def_handle, object_root_trans_init)
        else:
            object_def_handle = env_def.get_rigid_body_def_handle_by_name(object_name)
            object_handle = env_def.create_rigid_body(object_def_handle, object_root_trans_init, object_name)

        self.object_handles[object_name] = object_handle
        self._init_object_transforms[object_name] = init_transform

        # Keep legacy attributes to avoid breaking downstream code.
        setattr(self, f"{object_name}_handle", object_handle)
        setattr(self, f"_init_{object_name}_transforms", init_transform)

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
        self.robot_6d_robot_to_world = torch.zeros((self.total_num_envs, 6), device=self.device, dtype=torch.float32)
        self.robot_palm_down_in_robot = (
            torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).unsqueeze(0).repeat(self.total_num_envs, 1)
        )
        self.robot_palm_down_in_world = torch.zeros((self.total_num_envs, 3), device=self.device, dtype=torch.float32)
        self.robot_linear_velocity_in_world = torch.zeros((self.total_num_envs, 3), device=self.device, dtype=torch.float32)
        self.robot_angular_velocity_in_world = torch.zeros((self.total_num_envs, 3), device=self.device, dtype=torch.float32)

        # Per-object readback state buffers (all objects).
        self.get_object_pos_bufs: Dict[str, torch.Tensor] = {}
        self.get_object_vel_bufs: Dict[str, torch.Tensor] = {}
        for object_name in self.object_creation_order:
            self.get_object_pos_bufs[object_name] = torch.zeros((self.total_num_envs, 7), device=self.device, dtype=torch.float32)
            self.get_object_vel_bufs[object_name] = torch.zeros((self.total_num_envs, 6), device=self.device, dtype=torch.float32)

        # Selected reward object read/write buffers.
        self.get_reward_object_pos_buf = self.get_object_pos_bufs[self.reward_object]
        self.get_reward_object_vel_buf = self.get_object_vel_bufs[self.reward_object]
        self.set_reward_object_pos_buf = torch.zeros((self.total_num_envs, 7), device=self.device, dtype=torch.float32)
        self.set_reward_object_vel_buf = torch.zeros((self.total_num_envs, 6), device=self.device, dtype=torch.float32)

        self.gpu_init_reward_object_transforms = torch.tensor(self._init_reward_object_transforms, dtype=torch.float32, device=self.device)
        self.gpu_init_reward_object_velocities = torch.zeros(6, dtype=torch.float32, device=self.device)

        self.set_reward_object_pos_buf[:] = self.gpu_init_reward_object_transforms
        self.set_reward_object_vel_buf[:] = self.gpu_init_reward_object_velocities

        # Goal reward object rotation.
        self.object_goal_pos_in_world = torch.zeros((self.total_num_envs, 3), device=self.device, dtype=torch.float32)
        self.object_goal_quat_object_to_world = (
            torch.tensor([0, 0, 0, 1], device=self.device, dtype=torch.float32).unsqueeze(0).repeat(self.total_num_envs, 1)
        )
        self.object_goal_6d_object_to_world = quaternion_to_6d(self.object_goal_quat_object_to_world)
        self.object_in_hand_goal_6d_object_to_robot = quaternion_to_6d(self.object_goal_quat_object_to_world)
        if self.reward_object == "pen":
            self.object_down_in_object = (
                torch.tensor([1, 0, 0], device=self.device, dtype=torch.float32).unsqueeze(0).repeat(self.reset_buf.sum(), 1)
            )
        else:
            self.object_down_in_object = (
                torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).unsqueeze(0).repeat(self.reset_buf.sum(), 1)
            )
        self.object_position_in_world = torch.zeros((self.total_num_envs, 3), device=self.device, dtype=torch.float32)
        self.object_quat_object_to_world = torch.zeros((self.total_num_envs, 4), device=self.device, dtype=torch.float32)
        self.object_6d_object_to_world = torch.zeros((self.total_num_envs, 6), device=self.device, dtype=torch.float32)
        self.object_down_in_world = torch.zeros((self.total_num_envs, 3), device=self.device, dtype=torch.float32)
        self.object_down_in_robot = torch.zeros((self.total_num_envs, 3), device=self.device, dtype=torch.float32)
        self.object_linear_velocity_in_world = torch.zeros((self.total_num_envs, 3), device=self.device, dtype=torch.float32)
        self.object_angular_velocity_in_world = torch.zeros((self.total_num_envs, 3), device=self.device, dtype=torch.float32)
        self.object_position_in_robot_frame = torch.zeros((self.total_num_envs, 3), device=self.device, dtype=torch.float32)
        self.object_orientation_in_robot_frame = torch.zeros((self.total_num_envs, 4), device=self.device, dtype=torch.float32)
        self.object_linear_velocity_in_robot_frame = torch.zeros((self.total_num_envs, 3), device=self.device, dtype=torch.float32)
        self.object_angular_velocity_in_robot_frame = torch.zeros((self.total_num_envs, 3), device=self.device, dtype=torch.float32)

        self.object_position_in_world_by_name: Dict[str, torch.Tensor] = {}
        self.object_quat_object_to_world_by_name: Dict[str, torch.Tensor] = {}
        self.object_linear_velocity_in_world_by_name: Dict[str, torch.Tensor] = {}
        self.object_angular_velocity_in_world_by_name: Dict[str, torch.Tensor] = {}
        for object_name in self.object_creation_order:
            self.object_position_in_world_by_name[object_name] = torch.zeros(
                (self.total_num_envs, 3), device=self.device, dtype=torch.float32
            )
            self.object_quat_object_to_world_by_name[object_name] = torch.zeros(
                (self.total_num_envs, 4), device=self.device, dtype=torch.float32
            )
            self.object_linear_velocity_in_world_by_name[object_name] = torch.zeros(
                (self.total_num_envs, 3), device=self.device, dtype=torch.float32
            )
            self.object_angular_velocity_in_world_by_name[object_name] = torch.zeros(
                (self.total_num_envs, 3), device=self.device, dtype=torch.float32
            )

        self.object_goal_down_in_world = torch.zeros((self.total_num_envs, 3), device=self.device, dtype=torch.float32)
        self.object_in_hand_goal_down_in_robot = torch.zeros((self.total_num_envs, 3), device=self.device, dtype=torch.float32)

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

        # Contact query buffers for reward-object-hand contact reward
        self.contact_normals_buf = torch.zeros(
            (self.max_contact_pairs_per_env * self.total_num_envs, 3), dtype=torch.float32, device=self.device
        )
        self.contact_point_seps_buf = torch.zeros(
            (self.max_contact_pairs_per_env * self.total_num_envs, 4), dtype=torch.float32, device=self.device
        )
        self.contact_id_a_buf = torch.zeros(
            (self.max_contact_pairs_per_env * self.total_num_envs, 4), dtype=torch.uint32, device=self.device
        )
        self.contact_id_b_buf = torch.zeros(
            (self.max_contact_pairs_per_env * self.total_num_envs, 4), dtype=torch.uint32, device=self.device
        )

        # Contacts
        self.object_hand_contact_buf = torch.zeros((self.total_num_envs,), device=self.device, dtype=torch.float32)
        self.object_hand_contact_count_buf = torch.zeros((self.total_num_envs,), device=self.device, dtype=torch.float32)

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

        ### Object commands
        # Read state for all objects.
        get_rigid_object_kin_cmds = []
        get_deformable_object_kin_cmds = []
        for object_name in self.object_creation_order:
            if self.object_is_deformable[object_name]:
                get_deformable_object_kin_cmds.append(
                    self.env_group.create_deformable_kinematic_state_command(
                        v.wrap_gpu_buffer(self.get_object_pos_bufs[object_name]),
                        v.wrap_gpu_buffer(self.get_object_vel_bufs[object_name]),
                        self.object_handles[object_name],
                    )
                )
            else:
                get_rigid_object_kin_cmds.append(
                    self.env_group.create_rigid_body_kinematic_state_command(
                        v.wrap_gpu_buffer(self.get_object_pos_bufs[object_name]),
                        v.wrap_gpu_buffer(self.get_object_vel_bufs[object_name]),
                        self.object_handles[object_name],
                    )
                )

        self.gpu_get_rigid_object_kin_cmd_array = (
            self.gym.create_gpu_array(get_rigid_object_kin_cmds) if len(get_rigid_object_kin_cmds) > 0 else None
        )
        self.gpu_get_deformable_object_kin_cmd_array = (
            self.gym.create_gpu_array(get_deformable_object_kin_cmds) if len(get_deformable_object_kin_cmds) > 0 else None
        )

        # Set selected reward object state (for reset).
        if self.reward_object_is_deformable:
            set_reward_object_kin_cmd = self.env_group.create_deformable_kinematic_state_command(
                v.wrap_gpu_buffer(self.set_reward_object_pos_buf),
                v.wrap_gpu_buffer(self.set_reward_object_vel_buf),
                self.reward_object_handle,
                masks_buffer=v.wrap_gpu_buffer(self.reset_buf),
            )
        else:
            set_reward_object_kin_cmd = self.env_group.create_rigid_body_kinematic_state_command(
                v.wrap_gpu_buffer(self.set_reward_object_pos_buf),
                v.wrap_gpu_buffer(self.set_reward_object_vel_buf),
                self.reward_object_handle,
                masks_buffer=v.wrap_gpu_buffer(self.reset_buf),
            )
        self.gpu_set_reward_object_kin_cmd_array = self.gym.create_gpu_array([set_reward_object_kin_cmd])

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

    def visualize_goal(self):
        # return
        if self.gym.get_render() is not None:
            goal_pos = self.object_goal_pos_in_world[0]
            goal_quat = self.object_goal_quat_object_to_world[0:1]
            goal_axes = [
                rotate_by_quat_A_to_B(
                    goal_quat,
                    torch.tensor([[1.0, 0.0, 0.0]], device=self.device, dtype=torch.float32),
                )[0],
                rotate_by_quat_A_to_B(
                    goal_quat,
                    torch.tensor([[0.0, 1.0, 0.0]], device=self.device, dtype=torch.float32),
                )[0],
                rotate_by_quat_A_to_B(
                    goal_quat,
                    torch.tensor([[0.0, 0.0, 1.0]], device=self.device, dtype=torch.float32),
                )[0],
            ]
            goal_points = [
                [
                    v.Vec3(goal_pos[0].item(), goal_pos[1].item(), goal_pos[2].item()),
                    v.Vec3(
                        (goal_pos[0] + 0.1 * goal_axes[0][0]).item(),
                        (goal_pos[1] + 0.1 * goal_axes[0][1]).item(),
                        (goal_pos[2] + 0.1 * goal_axes[0][2]).item(),
                    ),
                ],
                [
                    v.Vec3(goal_pos[0].item(), goal_pos[1].item(), goal_pos[2].item()),
                    v.Vec3(
                        (goal_pos[0] + 0.1 * goal_axes[1][0]).item(),
                        (goal_pos[1] + 0.1 * goal_axes[1][1]).item(),
                        (goal_pos[2] + 0.1 * goal_axes[1][2]).item(),
                    ),
                ],
                [
                    v.Vec3(goal_pos[0].item(), goal_pos[1].item(), goal_pos[2].item()),
                    v.Vec3(
                        (goal_pos[0] + 0.1 * goal_axes[2][0]).item(),
                        (goal_pos[1] + 0.1 * goal_axes[2][1]).item(),
                        (goal_pos[2] + 0.1 * goal_axes[2][2]).item(),
                    ),
                ],
            ]

            goal_transform = v.Transform(
                v.Quat(
                    self.object_goal_quat_object_to_world[0, 0].item(),
                    self.object_goal_quat_object_to_world[0, 1].item(),
                    self.object_goal_quat_object_to_world[0, 2].item(),
                    self.object_goal_quat_object_to_world[0, 3].item(),
                ),
                v.Vec3(
                    self.object_goal_pos_in_world[0, 0].item(),
                    self.object_goal_pos_in_world[0, 1].item(),
                    self.object_goal_pos_in_world[0, 2].item(),
                ),
            )

            # if self._cube is not None:
            #     self.gym.get_render().unregister_line_shape(self._cube)
            # self._cube = self.gym.get_render().create_user_line_cube(
            #     size=0.05,
            #     transform=goal_transform,
            #     color=v.Vec3(1, 0, 0),
            #     line_width=3.0,
            #     visible=True,
            #     env_handle=self.env_sets[0].get_environment_handle(0),
            # )
            # self.gym.get_render().register_line_shape(self._cube)

            for attr_name in ("_goal_axis_x", "_goal_axis_y", "_goal_axis_z"):
                if getattr(self, attr_name, None) is not None:
                    self.gym.get_render().unregister_line_shape(getattr(self, attr_name))

            self._goal_axis_x = self.gym.get_render().create_user_line(
                goal_points[0],
                v.Vec3(1.0, 0.0, 0.0),
                line_width=3.0,
                visible=True,
                env_handle=self.env_sets[0].get_environment_handle(0),
            )
            self._goal_axis_y = self.gym.get_render().create_user_line(
                goal_points[1],
                v.Vec3(0.0, 1.0, 0.0),
                line_width=3.0,
                visible=True,
                env_handle=self.env_sets[0].get_environment_handle(0),
            )
            self._goal_axis_z = self.gym.get_render().create_user_line(
                goal_points[2],
                v.Vec3(0.0, 0.0, 1.0),
                line_width=3.0,
                visible=True,
                env_handle=self.env_sets[0].get_environment_handle(0),
            )

            self.gym.get_render().register_line_shape(self._goal_axis_x)
            self.gym.get_render().register_line_shape(self._goal_axis_y)
            self.gym.get_render().register_line_shape(self._goal_axis_z)

    def resample_object_goal(self, reset_idx):
        # Sample position in world
        self.object_goal_pos_in_world[reset_idx, 0] = self.table_bounds[0, 0] + torch.rand(reset_idx.sum(), device=self.device) * (
            self.table_bounds[0, 1] - self.table_bounds[0, 0]
        )
        self.object_goal_pos_in_world[reset_idx, 1] = self.table_bounds[1, 0] + torch.rand(reset_idx.sum(), device=self.device) * (
            self.table_bounds[1, 1] - self.table_bounds[1, 0]
        )
        self.object_goal_pos_in_world[reset_idx, 2] = (
            self.table_bounds[2, 0]
            + 0.3
            + torch.rand(reset_idx.sum(), device=self.device) * (self.table_bounds[2, 1] - self.table_bounds[2, 0] - 0.3)
        )

        # Sample orientation in world
        self.object_goal_quat_object_to_world[reset_idx, :] = random_uniform_quaternion(
            reset_idx.sum(), device=self.device, dtype=torch.float32
        )
        self.object_goal_6d_object_to_world[reset_idx, :] = quaternion_to_6d(self.object_goal_quat_object_to_world[reset_idx, :])
        self.object_goal_down_in_world[reset_idx, :] = rotate_by_quat_A_to_B(
            self.object_goal_quat_object_to_world[reset_idx, :], self.object_down_in_object[reset_idx, :]
        )

        # Sample orientation in robot frame.
        object_in_hand_goal_quat_object_to_robot = random_uniform_quaternion(
            reset_idx.sum(), device=self.device, dtype=torch.float32
        )
        self.object_in_hand_goal_6d_object_to_robot[reset_idx, :] = quaternion_to_6d(object_in_hand_goal_quat_object_to_robot)
        # self.object_in_hand_goal_down_in_robot[reset_idx, :] = rotate_by_quat_A_to_B(
        #     object_in_hand_goal_quat_object_to_robot, self.object_down_in_object[reset_idx, :]
        # )
        self.object_in_hand_goal_down_in_robot[reset_idx, :] = (
            torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).unsqueeze(0).repeat(reset_idx.sum(), 1)
        )
        self.visualize_goal()

    def reset_idx(self):
        self.act_buf[self.reset_buf, :] = 0.0
        self.last_act_buf[self.reset_buf, :] = 0.0

        # Reset Hand Kinematics
        grasp = torch.rand((self.reset_buf.sum(), 1), device=self.device) * 0.5 * torch.pi
        per_finger = torch.ones((self.reset_buf.sum(), self.num_joints), device=self.device)
        self.reset_joint_pos_buf[self.reset_buf, :] = grasp * per_finger
        # self.reset_joint_pos_buf[self.reset_buf, :] = (
        #     torch.rand((self.reset_buf.sum(), self.num_joints), device=self.device) * 0.5 * torch.pi
        # )
        # self.reset_joint_pos_buf[self.reset_buf, :] = 0.0
        self.reset_joint_vel_buf[self.reset_buf, :] = 0.0
        self.reset_root_transform_buf[self.reset_buf, :4] = (
            torch.tensor([0.707, 0, 0, 0.707], device=self.device, dtype=torch.float32).unsqueeze(0).repeat(self.reset_buf.sum(), 1)
        )
        # self.reset_root_transform_buf[self.reset_buf, :4] = random_uniform_quaternion(
        #     self.reset_buf.sum(), device=self.device, dtype=torch.float32
        # )
        # self.reset_root_transform_buf[self.reset_buf, 4:] = (
        #     self.table_bounds[:, 0]
        #     + 0.1
        #     + torch.rand((self.reset_buf.sum(), 3), device=self.device) * (self.table_bounds[:, 1] - self.table_bounds[:, 0] - 0.1)
        # )
        self.reset_root_transform_buf[self.reset_buf, 4:] = torch.tensor([0.0, 0.0, 0.15], device=self.device)
        self.reset_root_vel_buf[self.reset_buf, :] = 0.0
        self.gym.set_articulation_kinematic_states(self.gpu_reset_kinematic_state_command_array)

        # Reset selected reward object state with noise.
        self.set_reward_object_pos_buf[self.reset_buf, :4] = random_uniform_quaternion(
            self.reset_buf.sum(), device=self.device, dtype=torch.float32
        )
        # self.set_reward_object_pos_buf[self.reset_buf, 4:] = (
        #     self.table_bounds[:, 0]
        #     + 0.1
        #     + torch.rand((self.reset_buf.sum(), 3), device=self.device) * (self.table_bounds[:, 1] - self.table_bounds[:, 0] - 0.1)
        # )
        # self.set_reward_object_pos_buf[self.reset_buf, 4:6] = self.reset_root_transform_buf[self.reset_buf, 4:6]
        # self.set_reward_object_pos_buf[self.reset_buf, 6] += (
        #     torch.rand((self.reset_buf.sum(),), device=self.device) * 0.05 + 0.05
        # )  # ensure object is above the hand
        self.set_reward_object_pos_buf[self.reset_buf, 4:] = torch.tensor([0.0, 0.0, 0.45], device=self.device)
        self.set_reward_object_vel_buf[self.reset_buf, :] = 0.0
        if self.reward_object_is_deformable:
            self.gym.set_deformable_kinematic_states(self.gpu_set_reward_object_kin_cmd_array)
        else:
            self.gym.set_rigid_body_kinematic_states(self.gpu_set_reward_object_kin_cmd_array)

        # Random goal
        update_indx = torch.rand(self.total_num_envs, device=self.device) < 0.005
        if update_indx.sum() > 0:
            self.resample_object_goal(update_indx)

        # Reset progress
        self.progress_buf[self.reset_buf] = 0

        # reset history
        self.obs_history.reset(self.reset_buf)

        # Randomize rigid body material
        # Write to buffers
        friction = torch.rand(len(self.num_envs), device=self.device) * 0.5 + 0.25  # [0.25, 0.75]
        self.set_static_friction_buf[:] = friction
        self.set_dynamic_friction_buf[:] = friction
        self.gym.set_rigid_material_properties(self.gpu_set_friction_cmd)

    def reset(self):
        obs, _ = super().reset()
        self.refresh_buffers()
        self.resample_object_goal(torch.ones_like(self.reset_buf, dtype=torch.bool))
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
        if self.thumb_motor_index >= 0:
            self.set_motor_cmd_buf[:, self.thumb_motor_index] = self.scaled_act_buf[:, 7]  # thumb command
        # self.set_motor_cmd_buf[:] = self.scaled_act_buf[:, 6:]
        self.gym.set_motor_forces(self.gpu_set_motor_control_command_array)

        # self.set_force_torque_buf[:] = 0.0
        # self.set_force_torque_buf[:, 0, :] = self.scaled_act_buf[:, :6]

        # Gravity compensation on base link
        self.set_force_torque_buf[:, :, 2] = 9.81 * self.link_masses
        self.gym.set_link_external_forces(self.set_force_torque_cmd_arr)

    def refresh_buffers(self):
        """Refresh all state buffers from simulation."""
        self.gym.get_articulation_kinematic_states(self.gpu_get_kinematic_state_command_array)
        if self.gpu_get_rigid_object_kin_cmd_array is not None:
            self.gym.get_rigid_body_kinematic_states(self.gpu_get_rigid_object_kin_cmd_array)
        if self.gpu_get_deformable_object_kin_cmd_array is not None:
            self.gym.get_deformable_kinematic_states(self.gpu_get_deformable_object_kin_cmd_array)

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
        self.robot_pos_in_world[:] = self.get_root_transform_buf[:, 4:7]
        self.robot_quat_robot_to_world[:] = self.get_root_transform_buf[:, 0:4]
        self.robot_6d_robot_to_world[:] = quaternion_to_6d(self.robot_quat_robot_to_world)
        self.robot_linear_velocity_in_world[:] = self.get_root_vel_buf[:, 3:6]
        self.robot_angular_velocity_in_world[:] = self.get_root_vel_buf[:, :3]

        self.robot_palm_down_in_world[:] = rotate_by_quat_A_to_B(self.robot_quat_robot_to_world, self.robot_palm_down_in_robot)

        # self.robot_angular_velocity_in_robot_frame = rotate_by_quat_A_to_B(quat_conjugate(self.robot_quat_robot_to_world), self.robot_angular_velocity_in_world)
        # self.robot_linear_velocity_in_robot_frame = rotate_by_quat_A_to_B(quat_conjugate(self.robot_quat_robot_to_world), self.robot_linear_velocity_in_world)

        # Read selected reward object state.
        self.object_position_in_world[:] = self.get_reward_object_pos_buf[:, 4:7]
        self.object_quat_object_to_world[:] = self.get_reward_object_pos_buf[:, 0:4]
        self.object_down_in_world[:] = rotate_by_quat_A_to_B(self.object_quat_object_to_world, self.object_down_in_object)
        self.object_down_in_robot[:] = rotate_by_quat_A_to_B(quat_conjugate(self.robot_quat_robot_to_world), self.object_down_in_world)
        self.object_6d_object_to_world[:] = quaternion_to_6d(self.object_quat_object_to_world)
        self.object_linear_velocity_in_world[:] = self.get_reward_object_vel_buf[:, 3:6]
        self.object_angular_velocity_in_world[:] = self.get_reward_object_vel_buf[:, :3]

        self.object_position_in_robot_frame[:] = rotate_by_quat_A_to_B(
            quat_conjugate(self.robot_quat_robot_to_world), self.get_reward_object_pos_buf[:, 4:7] - self.robot_pos_in_world
        )
        self.object_orientation_in_robot_frame[:] = quat_mul(
            quat_conjugate(self.robot_quat_robot_to_world), self.get_reward_object_pos_buf[:, 0:4]
        )
        self.object_linear_velocity_in_robot_frame[:] = rotate_by_quat_A_to_B(
            quat_conjugate(self.robot_quat_robot_to_world), self.get_reward_object_vel_buf[:, 3:6]
        )
        self.object_angular_velocity_in_robot_frame[:] = rotate_by_quat_A_to_B(
            quat_conjugate(self.robot_quat_robot_to_world), self.get_reward_object_vel_buf[:, :3]
        )

        # Read all object states for logging/analysis/debugging.
        for object_name in self.object_creation_order:
            self.object_position_in_world_by_name[object_name][:] = self.get_object_pos_bufs[object_name][:, 4:7]
            self.object_quat_object_to_world_by_name[object_name][:] = self.get_object_pos_bufs[object_name][:, 0:4]
            self.object_linear_velocity_in_world_by_name[object_name][:] = self.get_object_vel_bufs[object_name][:, 3:6]
            self.object_angular_velocity_in_world_by_name[object_name][:] = self.get_object_vel_bufs[object_name][:, :3]

        base_obs = torch.cat(
            [
                self.robot_pos_in_world,  # robot_position in world frame
                self.robot_palm_down_in_world,  # robot palm down vector in world frame
                self.robot_linear_velocity_in_world,  # Base link linear velocity in world frame
                self.robot_angular_velocity_in_world,  # Base link angular velocity in world frame
                # self.get_joint_pos_buf,  # Joint positions
                # self.get_joint_vel_buf,  # Joint velocities
                torch.mean(self.get_joint_pos_buf, dim=1, keepdim=True),
                (
                    self.get_joint_pos_buf[:, self.thumb_motor_index].view(-1, 1)
                    if self.thumb_motor_index is not None
                    else torch.zeros(self.total_num_envs, 1, device=self.device)
                ),  # Thumb joint position
                torch.mean(self.get_joint_vel_buf, dim=1, keepdim=True),
                (
                    self.get_joint_vel_buf[:, self.thumb_motor_index].view(-1, 1)
                    if self.thumb_motor_index is not None
                    else torch.zeros(self.total_num_envs, 1, device=self.device)
                ),  # Joint velocities
                self.act_buf,  # Current actions
                self.last_act_buf,  # Last actions
                self.object_position_in_world,
                self.object_down_in_world,
                self.object_linear_velocity_in_world,
                self.object_angular_velocity_in_world,
                self.object_goal_down_in_world,
                self.object_goal_pos_in_world,
                self.object_down_in_robot,
                self.object_in_hand_goal_down_in_robot,
            ],
            dim=-1,
        )

        self.object_hand_contact_buf[:], self.object_hand_contact_count_buf[:] = self._compute_object_hand_contact()
        self.obs_history.add(base_obs)

        self.obs_buf[:] = self.obs_history.get().view(self.total_num_envs, -1)

    def _compute_object_hand_contact(self) -> Tuple[torch.Tensor, torch.Tensor]:
        contact = torch.zeros(self.total_num_envs, dtype=torch.float32, device=self.device)
        contact_count = torch.zeros(self.total_num_envs, dtype=torch.float32, device=self.device)

        num_contacts = self.gym.get_rigid_contacts(
            v.wrap_gpu_buffer(self.contact_normals_buf),
            v.wrap_gpu_buffer(self.contact_point_seps_buf),
            v.wrap_gpu_buffer(self.contact_id_a_buf),
            v.wrap_gpu_buffer(self.contact_id_b_buf),
            self.max_contact_pairs_per_env * self.total_num_envs,
        )
        num_stored = min(num_contacts, self.max_contact_pairs_per_env * self.total_num_envs)
        if num_stored <= 0:
            return contact, contact_count

        id_a = self.contact_id_a_buf[:num_stored].to(torch.long)
        id_b = self.contact_id_b_buf[:num_stored].to(torch.long)

        env_a = self.contact_env_lookup[id_a[:, 1], id_a[:, 2]]
        env_b = self.contact_env_lookup[id_b[:, 1], id_b[:, 2]]
        same_env = env_a == env_b
        valid_env = torch.logical_and(env_a >= 0, env_b >= 0)
        valid_contact = torch.logical_and(same_env, valid_env)
        if not torch.any(valid_contact):
            return contact, contact_count

        env_indices = env_a.clamp_min(0)

        object_indices = self.reward_object_transform_index_by_env[env_indices]
        a_is_object = id_a[:, 3] == object_indices
        b_is_object = id_b[:, 3] == object_indices

        hand_indices = self.hand_transform_indices_by_env[env_indices]
        a_is_hand = torch.any(id_a[:, 3].unsqueeze(1) == hand_indices, dim=1)
        b_is_hand = torch.any(id_b[:, 3].unsqueeze(1) == hand_indices, dim=1)

        object_hand_contact = torch.logical_and(
            valid_contact,
            torch.logical_or(
                torch.logical_and(a_is_object, b_is_hand),
                torch.logical_and(b_is_object, a_is_hand),
            ),
        )

        if torch.any(object_hand_contact):
            contact_env_indices = env_indices[object_hand_contact]
            a_is_hand_contact = a_is_hand[object_hand_contact]
            hand_link_indices = torch.where(
                a_is_hand_contact,
                id_a[object_hand_contact, 3],
                id_b[object_hand_contact, 3],
            )

            # Count unique contacting hand links per environment.
            unique_env_link = torch.unique(contact_env_indices * self.num_links + hand_link_indices)
            unique_env_indices = unique_env_link // self.num_links
            touched_envs = torch.unique(unique_env_indices)

            contact[touched_envs] = 1.0
            contact_count.scatter_add_(
                0,
                unique_env_indices,
                torch.ones_like(unique_env_indices, dtype=torch.float32, device=self.device),
            )

        return contact, contact_count

    def compute_reward_termination_truncation(self):
        self.rew_buf[:] = 0.0

        # Reward object position tracking.
        object_pos_error = torch.norm(self.object_goal_pos_in_world - self.object_position_in_world, dim=-1)
        object_error_normalized = object_pos_error / 0.2
        object_pos_reward = torch.exp(-1.0 * object_error_normalized**2)
        self.info["rewards"]["pos_reward"] = object_pos_reward.sum().item() / self.total_num_envs
        self.rew_buf[:] += 1.0 * object_pos_reward

        # Reward upright orientation
        goal_alignment = torch.sum(self.object_goal_down_in_world * self.object_down_in_world, dim=-1)
        goal_alignment_clipped = torch.clamp(goal_alignment, max=1.0)
        goal_alignment_normalized = (goal_alignment_clipped + 1.0) / 2.0
        goal_alignment_reward = torch.exp(-3.0 * (1 - goal_alignment_normalized) ** 2)
        self.info["rewards"]["rot_reward"] = goal_alignment_reward.sum().item() / self.total_num_envs
        self.rew_buf[:] += 2.0 * goal_alignment_reward

        # Reward hand orientation alignment with object orientation.
        hand_goal_alignment = torch.sum(self.object_in_hand_goal_down_in_robot * self.object_down_in_robot, dim=-1)
        hand_goal_alignment_clipped = torch.clamp(hand_goal_alignment, max=1.0)
        hand_goal_alignment_normalized = (hand_goal_alignment_clipped + 1.0) / 2.0
        hand_goal_alignment_reward = torch.exp(-2.0 * (1 - hand_goal_alignment_normalized) ** 2)
        self.info["rewards"]["in_hand_rot_reward"] = hand_goal_alignment_reward.sum().item() / self.total_num_envs
        self.rew_buf[:] += 2.0 * hand_goal_alignment_reward

        # Reward distance between selected object and hand.
        dist = torch.norm(self.robot_pos_in_world - self.object_position_in_world, dim=-1)
        dist_clipped = torch.clamp(dist, min=0.05)  # don't reward getting too close to allow for exploration
        dist_clipped_normalized = dist_clipped / 0.2
        dist_rew = torch.exp(-1.0 * dist_clipped_normalized**2)
        self.info["rewards"]["dist_reward"] = dist_rew.sum().item() / self.total_num_envs
        self.rew_buf[:] += 1.0 * dist_rew

        # Reward robot alignment of palm with direction to selected object.
        hand_to_pen_dir_in_world = torch.nn.functional.normalize(self.object_position_in_world - self.robot_pos_in_world, dim=-1)
        palm_down_in_world = palm_down_in_world_frame_from_quat_robot_to_world(self.get_root_transform_buf[:, 0:4])
        alignment = torch.sum(
            hand_to_pen_dir_in_world * palm_down_in_world, dim=-1
        )  # range [-1, 1], where 1 means palm is perfectly aligned with direction to object
        alignment_clipped = torch.clamp(alignment, max=0.0)  # don't reward perfect alignment to allow for exploration
        aligment_normalized = (alignment_clipped + 1.0) / 2.0  # scale to [0, 1]
        alignment_rew = torch.exp(-1.0 * (1 - aligment_normalized) ** 2)
        self.info["rewards"]["alignment_reward"] = alignment_rew.sum().item() / self.total_num_envs
        self.rew_buf[:] += 1.0 * alignment_rew

        # Reward hand-object contacts.
        contact_reward = self.object_hand_contact_count_buf * 0.5
        self.info["rewards"]["contact_reward"] = contact_reward.sum().item() / self.total_num_envs
        self.rew_buf[:] += 0.5 * contact_reward

        # Penalize large actions to encourage smooth control
        action_penalty = torch.sum(self.act_buf**2, dim=-1)
        # action_penalty_normalized = action_penalty / self.num_actions
        # action_penalty_reward = torch.exp(-1.0 * action_penalty_normalized**2)
        action_penalty_reward = -1 * action_penalty
        self.info["rewards"]["action_penalty"] = action_penalty_reward.sum().item() / self.total_num_envs
        self.rew_buf[:] += 0.01 * action_penalty_reward

        # Action smoothness
        action_smoothness_penalty = torch.sum((self.act_buf - self.last_act_buf) ** 2, dim=-1)
        # action_smoothness_penalty_normalized = action_smoothness_penalty / self.num_actions
        # action_smoothness_reward = torch.exp(-1.0 * action_smoothness_penalty_normalized**2)
        action_smoothness_reward = -1 * action_smoothness_penalty
        self.info["rewards"]["action_smoothness_penalty"] = action_smoothness_reward.sum().item() / self.total_num_envs
        self.rew_buf[:] += 0.01 * action_smoothness_reward

        # Object velocity penalty to encourage more stable grasps
        object_linear_velocity_mag = torch.sum(self.object_linear_velocity_in_world**2, dim=-1)
        object_linear_velocity_penalty = -1 * object_linear_velocity_mag
        self.info["rewards"]["object_linear_velocity_penalty"] = object_linear_velocity_penalty.sum().item() / self.total_num_envs
        self.rew_buf[:] += 0.01 * object_linear_velocity_penalty

        object_angular_velocity_mag = torch.sum(self.object_angular_velocity_in_world**2, dim=-1)
        object_angular_velocity_penalty = -1 * object_angular_velocity_mag
        self.info["rewards"]["object_angular_velocity_penalty"] = object_angular_velocity_penalty.sum().item() / self.total_num_envs
        self.rew_buf[:] += 0.01 * object_angular_velocity_penalty

        # Penalize object drops and end episode
        # drop_penalty = self.object_position_in_world[:, 2] < -0.1
        drop_penalty = self.object_position_in_world[:, 2] < 0.225
        drop_reward = -1 * drop_penalty
        self.info["rewards"]["drop_penalty"] = drop_reward.sum().item() / self.total_num_envs
        self.rew_buf[:] += 1.0 * drop_reward

        # Penalize hand out of table bounds
        out_of_bounds = torch.logical_or(
            self.robot_pos_in_world < self.table_bounds[:, 0], self.robot_pos_in_world > self.table_bounds[:, 1]
        )
        bounds_penalty = torch.any(out_of_bounds, dim=-1)
        bounds_reward = -1 * bounds_penalty
        self.info["rewards"]["bounds_penalty"] = bounds_reward.sum().item() / self.total_num_envs
        self.rew_buf[:] += 1.0 * bounds_reward

        # Terminations
        self.term_buf[:] = torch.logical_or(drop_penalty, bounds_penalty)
        # self.term_buf[:] = drop_penalty

        # Truncations
        self.trunc_buf[:] = self.progress_buf >= self.max_episode_length


if __name__ == "__main__":
    from vlearn.utils import get_VL_VISUAL_TESTS

    # Configuration
    config = {
        "num_envs": 1,
        "rendering": True,
        "with_window": get_VL_VISUAL_TESTS(),
        "max_episode_length": 50 * 60,
        "fixed_hand": True,
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

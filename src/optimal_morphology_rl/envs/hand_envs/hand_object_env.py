import math
import torch
import numpy as np
from vlearn.spaces import Box
import vlearn as v
import os
import sys
from optimal_morphology_rl.modules.contacts import Contacts
from optimal_morphology_rl.modules.kinematic_sensor import KinematicSensor
from optimal_morphology_rl.modules.force_sensors import ForceSensors
from optimal_morphology_rl.modules.robot import Robot
from optimal_morphology_rl.modules.object_generator import ObjectGenerator, LoadedRigidObject
from optimal_morphology_rl.modules.object_camera_recorder import ObjectCameraRecorder
from optimal_morphology_rl.modules.external_force import ExternalForceConfig, ExternalForceModule

from pathlib import Path

# TODO: Refactor to avoid this hack to import from the vlearn repo.
sys.path.append(os.path.join(os.path.dirname(__file__), "/workspace/vlearn/train/envs/"))
from environment import EnvironmentGpu

from time_series_buffer.time_series_buffer import TimeSeriesBuffer
from optimal_morphology_rl.envs.hand_envs.helpers.hand_pen_helpers import (
    rotate_by_quat_A_to_B,
)
from optimal_morphology_rl.helpers.numpy_vlearn import quaternion_to_6d

from vlearn.torch_utils.torch_jit_utils import scale


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
        max_episode_length: int = 6 * 60,  # 6 seconds at 60Hz control frequency
        gravity: v.Vec3 = v.Vec3(0, 0, -9.81),
        timestep: float = 1 / 120,  # 120Hz sim frequency
        frame_skip: int = 2,  # 60Hz control step
        spacing: float = 0.5,
        initial_is_paused: bool = False,
        raise_exception: bool = False,
        print_hash: bool = False,
        force_mass_inertia_computation: bool = False,
        with_window: bool = True,
        fixed_hand: bool = False,
        vsim_path: str = None,
        record_output_path: str = None,
        object: str = "drawer",
    ):

        self.max_contact_pairs_per_env = 64
        self.max_contact_patches_per_env = -1
        self.max_contact_points_per_patch = 4

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
            initial_render_substep=False,
            raise_exception=raise_exception,
            up_axis=v.Vec3(0, 0, 1),
            print_hash=print_hash,
            with_window=with_window,
            max_contact_pairs_per_env=self.max_contact_pairs_per_env,
            max_contact_patches_per_env=self.max_contact_patches_per_env,
            max_contact_points_per_patch=self.max_contact_points_per_patch,
            treat_warning_as_error=True,
        )

        self.num_envs_per_set = 1
        if self.num_envs % self.num_envs_per_set != 0:
            raise ValueError(f"num_envs must be a multiple of {self.num_envs_per_set}.")
        self.num_envs = [self.num_envs_per_set] * (self.num_envs // self.num_envs_per_set)
        self.device = device
        self.max_episode_length = max_episode_length
        self.reward_object_name: str = object
        self.fixed_hand = True if object == "cube" else False
        self.num_hist = 3
        self.hist_stride = 10
        self.obs_history_length = 1 + (self.num_hist - 1) * self.hist_stride

        # Initialize ObjectGenerator with object names
        self.objects = ObjectGenerator(
            object_names=[self.reward_object_name, "table" if record_output_path is None else "table_with_camera"]
        )
        self.reward_object = self.objects.get_object(self.reward_object_name)
        self.camera = ObjectCameraRecorder(record_output_path) if record_output_path is not None else None
        self.forces = ForceSensors()
        self.kinematic_sensor = KinematicSensor()

        # Create environments
        self.create_envs(vsim_path)

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

        reward_object_link_offset = self.objects.get_object_link_offset(self.reward_object.name)
        self.contacts = Contacts(self, reward_object_link_offset, link_names=["distal"])
        if isinstance(self.reward_object, LoadedRigidObject):
            self.force_module = ExternalForceModule(
                body_handles={object: self.reward_object.handle},
                total_num_envs=num_envs,
                device=device,
                env_group=self.env_group,
                gym=self.gym,
                config=ExternalForceConfig(),
            )
        else:
            self.force_module = None

        if self.gym.get_render() is not None:
            self.gym_render.reset_camera(v.Vec3(-0.671139, 0.073098, 0.726423), v.Vec3(0.755459, -0.009100, -0.655133))

        self.info["rewards"] = {}

    def create_envs(self, vsim_path):
        """Create simulation environments."""
        # Create environment definition
        self.env_def_handle = self.gym.create_environment_def("hand_env")
        env_def = self.gym.get_environment_def(self.env_def_handle)

        self.robot = Robot(fixed_hand = self.fixed_hand, use_tendon=True)
        self.robot.create_envs(env_def, vsim_path, self.device)

        # Load all objects through ObjectGenerator (includes table)
        self.objects.load(env_def)

        env_set_offsets = []
        spacing = math.sqrt(self.num_envs_per_set)
        for i in range(len(self.num_envs)):
            # set x and y into square grid:
            grid_size = math.ceil(math.sqrt(len(self.num_envs)))
            x = (i % grid_size) * spacing
            y = (i // grid_size) * spacing
            env_set_offsets.append(v.Vec3(x, y, 0))

        if self.camera is not None:
            self.camera.build_specs(self.objects, env_def)

        env_def.finalize()
        super().create_envs(self.env_def_handle, env_set_offsets=env_set_offsets)

        if self.camera is not None:
            self.camera.build_cameras(env_def, self.env_group, self.gym, self.num_envs, self.device)

    def _setup_action_space(self):
        """Configure action space dimensions."""
        self.num_actions = 6 + self.robot.get_num_dofs()  # wrist velocities (6) + motor commands

        self.action_space = Box(
            low=np.full(self.num_actions, -1.0, dtype=np.float32), high=np.full(self.num_actions, 1.0, dtype=np.float32), dtype=np.float32
        )

        self.velocity_scale = torch.tensor([1.0, 1.0, 1.0, 0.2, 0.2, 0.2], dtype=torch.float32, device=self.device)
        self.max_velocity = self.velocity_scale * 2.0

        min_scale = -1.0 * self.robot.max_torque
        max_scale = 1.0 * self.robot.max_torque
        if self.robot.use_tendon:
            min_scale = -0.25 * self.robot.tendon_max_force
            max_scale = 1.0 * self.robot.tendon_max_force

        self.min_revolute_scale = torch.full((self.robot.get_num_dofs(),), min_scale, device=self.device)
        self.max_revolute_scale = torch.full((self.robot.get_num_dofs(),), max_scale, device=self.device)

    def _setup_observation_space(self):
        """Configure observation space dimensions."""
        self.base_obs_slices = {}
        obs_offset = 0
        for name, width in [
            ("robot_pos_in_world", 3),
            ("_6d_robot_to_world", 6),
            ("robot_linear_velocity_in_world", 3),
            ("robot_angular_velocity_in_world", 3),
            ("get_joint_pos_buf", self.robot.num_joints),
            ("get_joint_vel_buf", self.robot.num_joints),
            ("act_buf", self.num_actions),
            ("object_position_in_world", 3),
            ("_6d_object_to_world", 6),
            ("object_linear_velocity_in_world", 3),
            ("object_angular_velocity_in_world", 3),
            ("object_goal_pos_in_world", 3),
            ("_6d_object_goal_to_world", 6),
            # ("trajectory_timestep", 1),
        ]:
            self.base_obs_slices[name] = slice(obs_offset, obs_offset + width)
            obs_offset += width

        self.base_num_obs = obs_offset

        # Joint positions + velocities + ball state + actions
        self.num_obs = self.base_num_obs * self.num_hist

        # Observation history buffer.
        self.base_obs = torch.zeros((self.total_num_envs, self.base_num_obs), device=self.device, dtype=torch.float32)
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

    def allocate_buffers(self):
        """Allocate GPU buffers for state and control."""
        super().allocate_buffers()

        ### Env Buffers
        self.inverse_reset_buf = torch.zeros(self.total_num_envs, device=self.device, dtype=torch.bool)

        self.robot.allocate_buffers(self.total_num_envs, self.device)

        # Allocate object buffers through ObjectGenerator
        self.objects.allocate_buffers(self.total_num_envs, self.device)

        env_def = self.gym.get_environment_def(self.env_def_handle)
        articulation = env_def.get_articulation(self.robot.arti_handle)
        self.kinematic_sensor.allocate_buffers(env_def, self.reward_object.handle, self.total_num_envs, self.device)
        self.forces.allocate_buffers(self.robot.art_def, articulation, self.total_num_envs, self.device, link_names=["distal"])

        # Setup table bounds from table object
        table_obj = (
            self.objects.get_object("table")
            if self.objects.get_object("table") is not None
            else self.objects.get_object("table_with_camera")
        )
        self.table_half_size = table_obj.half_size
        self.table_bounds = torch.tensor(
            [
                [-self.table_half_size.x, self.table_half_size.x],
                [-self.table_half_size.y, self.table_half_size.y],
                [0.0, 0.35],
            ],
            device=self.device,
            dtype=torch.float32,
        )

        # Goal buffers remain env-owned.
        self._6d_object_goal_to_world = torch.zeros((self.total_num_envs, 6), device=self.device, dtype=torch.float32)

        # Last action buffer
        self.last_act_buf = torch.zeros_like(self.act_buf)
        self.scaled_act_buf = torch.zeros_like(self.act_buf)

        self.timestep_buf = torch.zeros((self.total_num_envs,), device=self.device, dtype=torch.long)

    def create_gpu_commands(self):
        """Create GPU command arrays for efficient state queries and control."""
        self.robot.create_gpu_commands(self.env_group, self.gym, self.reset_buf, self.inverse_reset_buf)

        ### Object commands - managed by ObjectGenerator
        self.objects.create_gpu_commands(self.env_group, self.gym, self.reset_buf)

        self.kinematic_sensor.create_gpu_commands(self.env_group, self.gym)
        self.forces.create_gpu_commands(self.env_group, self.gym)

    def visualize_goal(self):
        if self.gym.get_render() is None:
            return

        goal_pos = self.reward_object.goal_pos_in_world[0]
        goal_quat = self.reward_object.goal_quat_object_to_world[0:1]
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

    def reset_idx(self):
        if self.reset_buf.sum() == 0:
            return
        
        # Reset environment buffers
        self.act_buf[self.reset_buf, :] = 0.0
        self.last_act_buf[self.reset_buf, :] = 0.0
        self.progress_buf[self.reset_buf] = 0
        self.timestep_buf[self.reset_buf] = 0
        self.obs_history.reset_idx(self.reset_buf)

        # Reset modules
        self.robot.reset_idx(self.gym, self.reset_buf, self.device)
        self.reward_object.reset_idx(self.gym, self.reset_buf)
        self.visualize_goal()

    def reset(self):
        obs, _ = super().reset()
        self.refresh_buffers()
        if self.total_num_envs != 1:  # when testing we prefer to start from 0
            self.progress_buf[:] = torch.randint(0, self.max_episode_length, (self.total_num_envs,), device=self.device)
        return obs, {}

    def pre_physics_step(self, actions: torch.Tensor):
        self.last_act_buf[:] = self.act_buf[:]
        self.act_buf[:] = actions
        self.scaled_act_buf[:, :6] = scale(self.act_buf[:, :6], -self.velocity_scale, self.velocity_scale)
        self.scaled_act_buf[:, 6:] = scale(self.act_buf[:, 6:], self.min_revolute_scale, self.max_revolute_scale)

        self.robot.pre_physics_step(self.gym, self.scaled_act_buf, self.max_velocity)

        # Apply extral force
        if self.force_module is not None:
            self.force_module.step(self.gym)

    def refresh_buffers(self):
        """Refresh all state buffers from simulation."""
        self.robot.refresh_buffers(self.gym)
        self.objects.refresh_buffers(self.gym)
        if self.camera is not None:
            self.camera.update(self.gym)
            self.camera.save(self.timestep_buf[0].cpu().item())

    def post_physics_step(self):
        """Post-step processing."""
        self.progress_buf[:] += 1

        # Update timestep for expert tracking
        self.timestep_buf[:] += 1

        # Check for episode termination
        self.reset_buf[:] = torch.logical_or(self.term_buf, self.trunc_buf)
        self.inverse_reset_buf[:] = ~self.reset_buf
        self.reset_idx()

        # Update state
        self.refresh_buffers()
        self.compute_observations()
        self.compute_reward_termination_truncation()

        # Resample goal
        sample_new_goal = torch.rand(self.total_num_envs, device=self.device) < 0.005
        if sample_new_goal.any():
            self.reward_object.update_goal(sample_new_goal)

    def compute_observations(self):
        """Construct observation vector."""
        self.kinematic_sensor.update(self.gym)

        robot_state = self.robot.get_state()

        object_pos_world = self.kinematic_sensor.pos_in_world
        object_quat_world = self.kinematic_sensor.quat_sensor_to_world
        object_lin_vel_world = self.kinematic_sensor.linear_velocity_world
        object_ang_vel_world = self.kinematic_sensor.angular_velocity_world
        object_6d_to_world = quaternion_to_6d(object_quat_world)
        object_goal_pos_in_world = self.reward_object.goal_pos_in_world
        quat_object_goal_to_world = self.reward_object.goal_quat_object_to_world
        self._6d_object_goal_to_world = quaternion_to_6d(quat_object_goal_to_world)

        self.base_obs[:, self.base_obs_slices["robot_pos_in_world"]] = robot_state["robot_pos_in_world"]
        self.base_obs[:, self.base_obs_slices["_6d_robot_to_world"]] = robot_state["_6d_robot_to_world"]
        self.base_obs[:, self.base_obs_slices["robot_linear_velocity_in_world"]] = robot_state["robot_linear_velocity_in_world"]
        self.base_obs[:, self.base_obs_slices["robot_angular_velocity_in_world"]] = robot_state["robot_angular_velocity_in_world"]
        self.base_obs[:, self.base_obs_slices["get_joint_pos_buf"]] = robot_state["get_joint_pos_buf"]
        self.base_obs[:, self.base_obs_slices["get_joint_vel_buf"]] = robot_state["get_joint_vel_buf"]
        self.base_obs[:, self.base_obs_slices["act_buf"]] = self.act_buf
        self.base_obs[:, self.base_obs_slices["object_position_in_world"]] = object_pos_world
        self.base_obs[:, self.base_obs_slices["_6d_object_to_world"]] = object_6d_to_world
        self.base_obs[:, self.base_obs_slices["object_linear_velocity_in_world"]] = object_lin_vel_world
        self.base_obs[:, self.base_obs_slices["object_angular_velocity_in_world"]] = object_ang_vel_world
        self.base_obs[:, self.base_obs_slices["object_goal_pos_in_world"]] = object_goal_pos_in_world
        self.base_obs[:, self.base_obs_slices["_6d_object_goal_to_world"]] = self._6d_object_goal_to_world

        self.obs_history.add(self.base_obs)

        self.obs_buf[:] = self.obs_history.get().view(self.total_num_envs, -1)

    def compute_reward_termination_truncation(self):
        self.rew_buf[:] = 0.0

        object_pos_in_world = self.kinematic_sensor.pos_in_world
        object_quat_world = self.kinematic_sensor.quat_sensor_to_world
        _6d_object_to_world = quaternion_to_6d(object_quat_world)

        # Reward for minimizing object-to-goal distance.
        obj_goal_dist = torch.norm(self.reward_object.goal_pos_in_world - object_pos_in_world, dim=-1)
        obj_goal_dist_normalized = obj_goal_dist / 0.2
        obj_goal_reward = torch.exp(-1.0 * obj_goal_dist_normalized**2)
        self.info["rewards"]["goal_position_reward"] = obj_goal_reward.sum().item() / self.total_num_envs
        self.info["rewards"]["goal_position_error_l2_norm_mm"] = obj_goal_dist.sum().item() / self.total_num_envs * 1000
        self.rew_buf[:] += 1.0 * obj_goal_reward

        # Reward upright orientation
        goal_alignment = torch.sum(self._6d_object_goal_to_world * _6d_object_to_world, dim=-1)
        goal_alignment_normalized = goal_alignment / 2.0
        goal_alignment_reward = goal_alignment_normalized
        self.info["rewards"]["goal_orientation"] = goal_alignment_reward.sum().item() / self.total_num_envs
        self.rew_buf[:] += 1.0 * goal_alignment_reward

        # Reward distance between selected object and hand.
        dist = torch.norm(self.robot.robot_pos_in_world - object_pos_in_world, dim=-1)
        dist_clipped = torch.clamp(dist, min=0.05)  # don't reward getting too close to allow for exploration
        dist_clipped_normalized = dist_clipped / 0.2
        dist_rew = torch.exp(-1.0 * dist_clipped_normalized**2)
        self.info["rewards"]["hand_to_object_distance"] = dist_rew.sum().item() / self.total_num_envs
        self.rew_buf[:] += 0.1 * dist_rew

        # Fingertip contact reward.
        self.contacts.update()  # biggest computational slowdown
        self.forces.update(self.gym)
        forces = self.forces.force_sensor_buf.norm(dim=-1)
        contacts = self.contacts.env_link_touch[:, self.contacts.monitored_link_mask]
        force_against_object = forces * contacts
        force_against_object.clamp_(0.0, 1.0)
        fingertip_contact_reward = torch.clamp(force_against_object.sum(dim=-1), min=0.0, max=3.0) / 3.0
        self.info["rewards"]["fingertip_contact_reward"] = fingertip_contact_reward.sum().item() / self.total_num_envs
        self.rew_buf[:] += 2.0 * fingertip_contact_reward

        # Penalize large actions to encourage smooth control
        action_penalty = torch.sum(self.act_buf**2, dim=-1)
        action_penalty_reward = -1 * action_penalty
        self.info["rewards"]["action_penalty"] = action_penalty_reward.sum().item() / self.total_num_envs
        self.rew_buf[:] += 0.01 * action_penalty_reward

        # Action smoothness
        action_smoothness_penalty = torch.sum((self.act_buf - self.last_act_buf) ** 2, dim=-1)
        action_smoothness_reward = -1 * action_smoothness_penalty
        self.info["rewards"]["action_smoothness_penalty"] = action_smoothness_reward.sum().item() / self.total_num_envs
        self.rew_buf[:] += 0.01 * action_smoothness_reward

        # Penalize object drops and end episode
        drop_penalty = object_pos_in_world[:, 2] < -0.1
        drop_reward = -1 * drop_penalty
        self.info["rewards"]["drop_penalty"] = drop_reward.sum().item() / self.total_num_envs
        self.rew_buf[:] += 10.0 * drop_reward

        # Penalize hand out of table bounds
        out_of_bounds = torch.logical_or(
            self.robot.robot_pos_in_world < self.table_bounds[:, 0], self.robot.robot_pos_in_world > self.table_bounds[:, 1]
        )
        bounds_penalty = torch.any(out_of_bounds, dim=-1)
        bounds_reward = -1 * bounds_penalty
        self.info["rewards"]["bounds_penalty"] = bounds_reward.sum().item() / self.total_num_envs
        self.rew_buf[:] += 10.0 * bounds_reward

        # Terminations
        self.term_buf[:] = torch.logical_or(drop_penalty, bounds_penalty)

        # Truncations
        self.trunc_buf[:] = self.progress_buf >= self.max_episode_length


async def main():
    from vlearn.utils import get_VL_VISUAL_TESTS

    # Configuration
    config = {
        "num_envs": 1,
        "rendering": True,
        "with_window": get_VL_VISUAL_TESTS(),
        "max_episode_length": 50 * 60,
        "fixed_hand": False,
    }

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda:0")

    lin_vel = torch.tensor([0.0, 0.0, 0.0], device=device)
    ang_vel = torch.tensor([0.0, 0.0, 0.0], device=device)
    grasp_force = torch.tensor([0.0] * 13, device=device)
    from optimal_morphology_rl.transfer.collector import HandController
    import asyncio

    controller = HandController(path="/dev/input/js0", linear_velocity=lin_vel, angular_velocity=ang_vel, grasp_force=grasp_force)

    loop = asyncio.get_event_loop()
    loop.create_task(controller.listen(timeout=30))

    # Create environment
    envs = HandObjectEnvironmentGpu(device=device, **config)
    obs, _ = envs.reset()

    gym = v.get_gym()
    render = gym.get_render()

    # Setup interactive control
    if render is not None:
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

    import numpy as np

    max_steps = 500
    training_data = np.zeros((max_steps, 41))

    # Main simulation loop
    print("\nStarting simulation...")
    step = 0
    while not envs.render_finished:
        # actions = control_fn()
        # print(grasp_force)
        # grasp_tensor = torch.ones((2,), device=device) * controller.grasp_force
        actions = torch.cat([ang_vel, lin_vel, grasp_force], dim=-1).unsqueeze(0).expand(config["num_envs"], -1)
        obs, rewards, terminations, truncations, infos = envs.step(actions)
        obs = envs.base_obs[:, :41]
        # print(obs[:, :6])
        training_data[step, :] = obs[0].cpu().numpy()
        step += 1
        if step >= max_steps:
            print("Saved data.")
            break
        await asyncio.sleep(0.01)  # Give event loop time to run

    # save numpy
    np.save("/workspace/optimal_morphology_rl/data/transfer/pick_up_example_41_new.npy", training_data)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
    # main()

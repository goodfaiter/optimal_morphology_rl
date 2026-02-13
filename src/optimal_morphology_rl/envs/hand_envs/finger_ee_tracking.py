import math
import torch
import numpy as np
from vlearn.spaces import Box
from typing import Tuple, List
import vlearn as v
import random
import wandb
from optimal_morphology_rl.tendon_model.buffer import TimeSeriesBuffer
from optimal_morphology_rl.tendon_model.tendon_force_estimation import TendonForceEstimation

# from tools.vlearn.train.envs.environment import EnvironmentGpu
import os
import sys

# sys.path.append(os.path.join(os.path.dirname(__file__), '/workspace/tools/vlearn/train/envs/'))
# from environment import EnvironmentGpu
from vlearn_train.envs.environment import EnvironmentGpu

# from vlearn.train.envs.environment import EnvironmentGpu
from vlearn.torch_utils.torch_jit_utils import scale, quat_mul, quat_conjugate, v_rpy_from_quat


class FingerEnvironmentGpu(EnvironmentGpu):
    """
    A simple tendon-force-control finger environment.
    The finger has 3 DOFs (revolute joints) and 3 tendons.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        rendering: bool = False,
        enable_scene_query: bool = False,
        max_episode_length: int = 20 * 20, # 20 seconds at 20Hz control frequency
        gravity: v.Vec3 = v.Vec3(0, 0, -9.81),
        timestep: float = 0.0125,  # 80Hz sim frequency
        frame_skip: int = 4,  # 20Hz control step
        spacing: float = 0.5,
        initial_is_paused: bool = False,
        send_interrupt: bool = False,
        print_hash: bool = False,
        max_contact_pairs_per_env: int = 64,
        has_self_collisions: bool = False,
        force_mass_inertia_computation: bool = False,
        with_window: bool = True,
        reset_noise_scale: float = 1.0,
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
            initial_render_substep=False,
            send_interrupt=send_interrupt,
            up_axis=v.Vec3(0, 0, 1),
            print_hash=print_hash,
            max_contact_pairs_per_env=max_contact_pairs_per_env,
            with_window=with_window,
        )

        self.num_envs_per_set = 32
        if self.num_envs % self.num_envs_per_set != 0:
            raise ValueError(f"num_envs must be a multiple of {self.num_envs_per_set}.")
        self.num_envs = [self.num_envs_per_set] * (self.num_envs // self.num_envs_per_set)
        self.device = device
        self.device_str = f"{self.device.type}:{self.device.index}"
        self.max_episode_length = max_episode_length
        self.has_self_collisions = has_self_collisions
        self.force_mass_inertia_computation = force_mass_inertia_computation
        self.reset_noise_scale = reset_noise_scale
        self._cube = None

        # Create environments
        self.create_envs()

        # Observation space
        self.hist_size = 1
        self.num_obs = 2 * self.num_tendons  # tendon lengths and tendon velocities
        self.num_obs += self.num_tendons  # action
        self.num_obs += self.num_tendons  # prev action
        self.num_obs += 1  # angle to goal
        self.num_obs *= self.hist_size  # history buffer
        print("num_obs: {}".format(self.num_obs))

        self.observation_space = Box(
            low=np.array([np.finfo("f").min] * self.num_obs, dtype=np.float32),
            high=np.array([np.finfo("f").max] * self.num_obs, dtype=np.float32),
            dtype=np.float32,
        )

        # Action space
        self.action_space = Box(
            low=np.array([-1] * self.num_tendons, dtype=np.float32),
            high=np.array([1] * self.num_tendons, dtype=np.float32),
            dtype=np.float32,
        )

        # Setup tendon joint mapping
        # self.setup_tendon_joint_mapping()

        # Allocate buffers
        self.allocate_buffers()

        # Create GPU commands
        self.create_gpu_commands()

        # Load model
        model_path = "/workspace/src/optimal_morphology_rl/tendon_model/2026_02_12_16_02_10_best.pt"
        self.tendon_force_model = TendonForceEstimation(model_path, num_envs=self.total_num_envs, device=self.device)

        # Finalize gym
        self.gym.gym_finalize()

        # Get default masses for randomization
        self.gym.get_link_properties(self.set_mass_cmd_arr)
        self.default_masses = torch.clone(self.set_mass_buf)

        # Randomize mass
        self.set_mass_buf[:] = self.default_masses + (torch.rand_like(self.default_masses) - 0.5) * 0.2 * self.default_masses
        self.gym.set_link_properties(self.set_mass_cmd_arr)

        # Randomize friction
        self.set_joint_friction_buf[:] = torch.rand((len(self.num_envs), self.num_dofs), device=self.device) * 0.1
        self.gym.set_joint_properties(self.set_joint_friction_cmd_arr)

        if self.gym.get_render() is not None:

            # Camera eye, dir: Vec3(-0.000000, -0.000000, 5.000000), Vec3(0.490521, 0.742927, -0.455465)
            # Camera eye, dir: Vec3(-0.125011, -0.001044, 0.065207), Vec3(0.880567, 0.397776, -0.257634)
            self.gym_render.reset_camera(v.Vec3(-0.125011, -0.001044, 0.065207), v.Vec3(0.880567, 0.397776, -0.257634))

        self.info['rewards'] = {}

    def create_envs(self):

        # Environment def
        self.env_def_handle = self.gym.create_environment_def("tendon_finger_env")
        env_def = self.gym.get_environment_def(self.env_def_handle)

        # Hand def
        hand_filename = "/workspace/assets/finger/ADAPT_Finger_v5.vsim"
        env_def.import_definitions(
            hand_filename,
            fixed=True,
            use_visual_mesh=True,
            force_mass_computation=self.force_mass_inertia_computation,
            force_inertia_computation=self.force_mass_inertia_computation,
        )
        self.art_def_handle = env_def.get_articulation_def_handle_by_name("ADAPT_Finger")
        self.art_def = env_def.get_articulation_def(self.art_def_handle)
        self.art_def.has_self_collisions = self.has_self_collisions
        if self.has_self_collisions:
            self.art_def.contact_offset = 0.005
        self.art_def.enable_control_type(v.ArticulationControlType.MOTOR, True)

        rot = v.shortest_rotation(v.Vec3(0, 0, 1), v.Vec3(0, 0, 1))
        pos = v.Vec3(0, 0, 0.1)
        self.root_trans_init = v.Transform(rot, pos)
        self.root_vel_init = v.SpatialVector(0)

        self.arti_handle = env_def.create_articulation(self.art_def_handle, self.root_trans_init, "ADAPT_Finger")

        self.num_dofs = self.art_def.get_num_joint_dof_defs()
        self.num_tendons = self.art_def.get_num_spatial_tendon_defs()
        self.num_links = self.art_def.get_num_link_defs()

        self.default_pully_radius = 0.0125
        self.default_zero_offset_length = 0.0740
        self.default_spring = 0.08
        self.pully_radius = torch.ones((self.total_num_envs, self.num_tendons), device=self.device) * self.default_pully_radius
        self.zero_offset_length = torch.ones((self.total_num_envs, self.num_tendons), device=self.device) * self.default_zero_offset_length
        self.spring = torch.ones((self.total_num_envs, self.num_dofs), device=self.device) * self.default_spring

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

        articulation = env_def.get_articulation(self.arti_handle)
        self.tip_sensor_handle = articulation.get_kinematic_sensor_handle(0)

    def visualize_goal(self):
        if self.gym.get_render() is not None:
            goal_transform = v.Transform(
                v.Quat(0, 0, 0, 1), v.Vec3(self.goal_pose[0, 0].item(), self.goal_pose[0, 1].item(), self.goal_pose[0, 2].item())
            )
            if self._cube is not None:
                self.gym.get_render().unregister_line_shape(self._cube)
            self._cube = self.gym.get_render().create_user_line_cube(
                size=0.005,
                transform=goal_transform,
                color=v.Vec3(1, 0, 0),
                line_width=3.0,
                visible=True,
                env_handle=self.env_sets[0].get_environment_handle(0),
            )
            self.gym.get_render().register_line_shape(self._cube)

    # def setup_tendon_joint_mapping(self):

    #     self.tendon_joint_mapping = {
    #         0: [0, 1],  # Tendon 0: both joints
    #     }

    #     # Convert to tensor for efficient computation
    #     self.tendon_joint_map = torch.zeros((self.num_tendons, self.num_dofs), dtype=torch.bool, device=self.device)
    #     for tendon_idx, joint_indices in self.tendon_joint_mapping.items():
    #         for joint_idx in joint_indices:
    #             self.tendon_joint_map[tendon_idx, joint_idx] = True

    #     print("\nTendon-Joint Mapping:")
    #     for tendon_idx, joint_indices in self.tendon_joint_mapping.items():
    #         if len(joint_indices) == 0:
    #             print(f"  Tendon {tendon_idx}: Free (no joints)")
    #         else:
    #             print(f"  Tendon {tendon_idx}: Joints {joint_indices}")

    def allocate_buffers(self):
        super().allocate_buffers()

        self.prev_act_buf = torch.zeros_like(self.act_buf)

        self.reset_dof_pos_buf = torch.zeros((self.total_num_envs, self.num_dofs), device=self.device, dtype=torch.float32)
        self.reset_dof_vel_buf = torch.zeros((self.total_num_envs, self.num_dofs), device=self.device, dtype=torch.float32)
        self.reset_root_transform_buf = torch.zeros((self.total_num_envs, 7), device=self.device, dtype=torch.float32)  # quat (4) + pos (3)
        self.reset_root_vel_buf = torch.zeros((self.total_num_envs, 6), device=self.device, dtype=torch.float32)

        # Reset spatial tendon buffer
        self.reset_tendon_length_buf = torch.zeros((self.total_num_envs, self.num_tendons), device=self.device, dtype=torch.float32)

        # Set joint force controls
        self.set_motor_cmd_buf = torch.zeros((self.total_num_envs, self.num_dofs), device=self.device, dtype=torch.float32)

        # Get joint states
        self.get_joint_pos_buf = torch.zeros((self.total_num_envs, self.num_dofs), device=self.device, dtype=torch.float32)

        # Set tendon controls
        self.set_tendon_controls_buf = torch.zeros((self.total_num_envs, self.num_tendons), device=self.device, dtype=torch.float32)

        # Get tendon states
        self.get_tendon_lengths_buf = torch.zeros((self.total_num_envs, self.num_tendons), device=self.device, dtype=torch.float32)
        self.get_tendon_vel_buf = torch.zeros((self.total_num_envs, self.num_tendons), device=self.device, dtype=torch.float32)

        # Goal
        self._angle = torch.zeros((self.total_num_envs, 1), device=self.device, dtype=torch.float32)
        self.goal_pose = torch.zeros((self.total_num_envs, 3), device=self.device, dtype=torch.float32)

        # Obs History Buffer
        self.motor_pos = TimeSeriesBuffer(num_envs=self.total_num_envs, max_size=self.hist_size, device=self.device)
        self.motor_vel = TimeSeriesBuffer(num_envs=self.total_num_envs, max_size=self.hist_size, device=self.device)
        self.act_history = TimeSeriesBuffer(num_envs=self.total_num_envs, max_size=self.hist_size, device=self.device)
        self.prev_act_history = TimeSeriesBuffer(num_envs=self.total_num_envs, max_size=self.hist_size, device=self.device)
        self.angle_history = TimeSeriesBuffer(num_envs=self.total_num_envs, max_size=self.hist_size, device=self.device)

    def create_gpu_commands(self):

        reset_kin_cmd = self.env_group.create_articulation_kinematic_state_command(
            v.wrap_gpu_buffer(self.reset_dof_pos_buf),
            v.wrap_gpu_buffer(self.reset_dof_vel_buf),
            v.wrap_gpu_buffer(self.reset_root_transform_buf),
            v.wrap_gpu_buffer(self.reset_root_vel_buf),
            self.arti_handle,
            (0, self.num_dofs),
            (0, 1),
            masks_buffer=v.wrap_gpu_buffer(self.reset_buf),
        )
        self.gpu_reset_kinematic_state_command_array = self.gym.create_articulation_kinematic_state_command_gpu_array([reset_kin_cmd])

        # Set motor/joint force control
        set_motor_cmd = self.env_group.create_motor_control_command(v.wrap_gpu_buffer(self.set_motor_cmd_buf), self.arti_handle)
        self.gpu_set_motor_control_command_array = self.gym.create_gpu_array([set_motor_cmd])

        # Get joint pos
        get_joint_positions_command = self.env_group.create_joint_state_command(v.wrap_gpu_buffer(self.get_joint_pos_buf), self.arti_handle)
        self.gpu_get_joint_positions_command_array = self.gym.create_gpu_array([get_joint_positions_command])

        # Set tendon control
        set_tendon_control_command = self.env_group.create_spatial_tendon_control_command(
            v.wrap_gpu_buffer(self.set_tendon_controls_buf), self.arti_handle
        )
        self.gpu_set_tendon_control_command_array = self.gym.create_gpu_array([set_tendon_control_command])

        # Get tendon states
        get_tendon_lengths_command = self.env_group.create_spatial_tendon_state_command(
            v.SpatialTendonState.LENGTH, v.wrap_gpu_buffer(self.get_tendon_lengths_buf), self.arti_handle, (0, self.num_tendons)
        )
        self.gpu_get_tendon_lengths_command_array = self.gym.create_gpu_array([get_tendon_lengths_command])

        get_tendon_velocities_command = self.env_group.create_spatial_tendon_state_command(
            v.SpatialTendonState.VELOCITY, v.wrap_gpu_buffer(self.get_tendon_vel_buf), self.arti_handle, (0, self.num_tendons)
        )
        self.gpu_get_tendon_velocities_command_array = self.gym.create_gpu_array([get_tendon_velocities_command])

        # Tip sensor
        self.get_tip_pose_sensor_buf = torch.zeros((self.total_num_envs, 7), dtype=torch.float32, device=self.device)
        self.get_tip_vel_sensor_buf = torch.zeros((self.total_num_envs, 6), dtype=torch.float32, device=self.device)

        get_tip_sensor_cmd = self.env_group.create_kinematic_sensor_state_command(
            v.wrap_gpu_buffer(self.get_tip_pose_sensor_buf),
            v.wrap_gpu_buffer(self.get_tip_vel_sensor_buf),
            self.tip_sensor_handle,
            frame_type=v.FrameType.ENVIRONMENT,
        )
        self.gpu_get_tip_sensor_command_array = self.gym.create_gpu_array([get_tip_sensor_cmd])

        # mass random
        # Create command
        self.set_mass_buf = torch.empty((len(self.num_envs), self.num_links), dtype=torch.float32, device=self.device)
        set_mass_cmd = self.env_group.create_link_property_command(
            v.LinkProperty.MASS,
            v.wrap_gpu_buffer(self.set_mass_buf),
            self.art_def_handle,
            (0, self.num_links),
            # v.wrap_gpu_buffer(self.reset_buf)
        )
        self.set_mass_cmd_arr = self.gym.create_gpu_array([set_mass_cmd])

        # Random fric
        # Create command
        self.set_joint_friction_buf = torch.empty((len(self.num_envs), self.num_dofs), dtype=torch.float32, device=self.device)
        set_joint_friction_cmd = self.env_group.create_joint_property_command(
            v.JointProperty.JOINT_FRICTION,
            v.wrap_gpu_buffer(self.set_joint_friction_buf),
            self.art_def_handle,
            (0, self.num_dofs),
            # masks_buffer=v.wrap_gpu_buffer(self.reset_buf),
        )

        self.set_joint_friction_cmd_arr = self.gym.create_gpu_array([set_joint_friction_cmd])

    def update_goal(self, update_indx):
        amp = 0.06
        self._angle[update_indx, :] = torch.rand((update_indx.sum(), 1), device=self.device) * 0.6 * torch.pi
        self.goal_pose[update_indx, 0] = 0
        self.goal_pose[update_indx, 1] = amp * torch.cos(self._angle[update_indx].squeeze()) + 0.03
        self.goal_pose[update_indx, 2] = amp * torch.sin(self._angle[update_indx].squeeze())
        self.visualize_goal()

    def reset_idx(self):
        self.act_buf[self.reset_buf, :] = 0.0

        self.reset_dof_pos_buf[self.reset_buf, :] = torch.rand((self.reset_buf.sum(), self.num_dofs), device=self.device) * 0.5 * torch.pi
        self.reset_dof_vel_buf[self.reset_buf, :] = 0.0
        self.reset_root_transform_buf[self.reset_buf, :4] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)  # identity quat
        self.reset_root_transform_buf[self.reset_buf, 4:] = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        self.reset_root_vel_buf[self.reset_buf, :] = 0.0
        self.gym.set_articulation_kinematic_states(self.gpu_reset_kinematic_state_command_array)

        self.gym.get_spatial_tendon_states(self.gpu_get_tendon_lengths_command_array)

        self.update_goal(self.reset_buf)

        # Reset progress
        self.progress_buf[self.reset_buf] = 0

        # reset buffers
        self.motor_pos.reset(self.reset_buf)
        self.motor_vel.reset(self.reset_buf)
        self.act_history.reset(self.reset_buf)
        self.prev_act_history.reset(self.reset_buf)
        self.angle_history.reset(self.reset_buf)
        
        # radomize radius, offset, and spring
        self.pully_radius[self.reset_buf, :] = self.default_pully_radius + (torch.rand((self.reset_buf.sum(), self.num_tendons), device=self.device) - 0.5) * 0.2 * self.default_pully_radius
        self.zero_offset_length[self.reset_buf, :] = self.default_zero_offset_length + (torch.rand((self.reset_buf.sum(), self.num_tendons), device=self.device) - 0.5) * 0.2 * self.default_zero_offset_length
        self.spring[self.reset_buf, :] = self.default_spring + (torch.rand((self.reset_buf.sum(), self.num_dofs), device=self.device) - 0.5) * 0.2 * self.default_spring

    def reset(self):
        return super().reset()

    def pre_physics_step(self, actions):
        self.prev_act_buf[:] = self.act_buf[:]
        self.act_buf[:] = actions[:]

        self.set_motor_cmd_buf[:] = self.get_joint_pos_buf[:, :] * -self.spring  # spring force
        self.gym.set_motor_forces(self.gpu_set_motor_control_command_array)

        motor_pos_rad = -1.0 * (self.get_tendon_lengths_buf - self.zero_offset_length) / self.pully_radius  # dL = radius [m] * angle [rad]
        des_pos_rad = torch.clamp(motor_pos_rad + self.act_buf, 0.0, 2.0 * torch.pi)
        motor_vel_rad_per_sec = -1.0 * self.get_tendon_vel_buf / self.pully_radius

        self.tendon_force_model.desired_position(des_pos_rad)
        self.tendon_force_model.measured_position(motor_pos_rad)
        self.tendon_force_model.measured_velocity(motor_vel_rad_per_sec)
        self.set_tendon_controls_buf[:] = self.tendon_force_model.forward().unsqueeze(1)
        self.set_tendon_controls_buf[:] = torch.clamp(self.set_tendon_controls_buf, 0.0, 30.0)
        self.gym.set_spatial_tendon_forces(self.gpu_set_tendon_control_command_array)

        # self.set_tendon_controls_buf[:] = self.act_buf[:]
        # self.gym.set_spatial_tendon_forces(self.gpu_set_tendon_control_command_array)

    def refresh_buffers(self):
        self.gym.get_joint_positions(self.gpu_get_joint_positions_command_array)
        self.gym.get_spatial_tendon_states(self.gpu_get_tendon_lengths_command_array)
        self.gym.get_spatial_tendon_states(self.gpu_get_tendon_velocities_command_array)
        self.gym.get_kinematic_sensor_states(self.gpu_get_tip_sensor_command_array)

    def post_physics_step(self):
        self.progress_buf[:] += 1

        self.reset_buf[:] = torch.logical_or(self.term_buf, self.trunc_buf)
        self.reset_idx()

        self.refresh_buffers()
        self.compute_observations()
        self.compute_reward_termination_truncation()

        # update goal angle
        update_indx = torch.rand(self.total_num_envs, device=self.device) < 0.025
        if update_indx.sum() > 0:
            self.update_goal(update_indx)

    def compute_observations(self):

        motor_pos_rad = -1.0 * (self.get_tendon_lengths_buf - self.zero_offset_length) / self.pully_radius
        motor_vel_rad_per_sec = -1.0 * self.get_tendon_vel_buf / self.pully_radius

        self.motor_pos.add(motor_pos_rad)
        self.motor_vel.add(motor_vel_rad_per_sec)
        self.act_history.add(self.act_buf[:, :])
        self.prev_act_history.add(self.prev_act_buf[:, :])
        self.angle_history.add(self._angle)

        # idx = 0
        # self.obs_buf[:, idx : idx + self.num_tendons * self.hist_size] = self.motor_pos.buffer[:, :]
        # idx += self.num_tendons * self.hist_size
        # self.obs_buf[:, idx : idx + self.num_tendons * self.hist_size] = self.motor_vel.buffer[:, :]
        # idx += self.num_tendons * self.hist_size
        # self.obs_buf[:, idx : idx + self.num_tendons * self.hist_size] = self.act_history.buffer[:, :]
        # idx += self.num_tendons * self.hist_size
        # self.obs_buf[:, idx : idx + self.num_tendons * self.hist_size] = self.prev_act_history.buffer[:, :]
        # idx += self.num_tendons * self.hist_size
        # self.obs_buf[:, idx : idx + self.hist_size] = self.angle_history.buffer[:, :]

        idx = 0
        self.obs_buf[:, idx : idx + self.num_tendons] = motor_pos_rad[:, :]
        idx += self.num_tendons
        self.obs_buf[:, idx : idx + self.num_tendons] = motor_vel_rad_per_sec[:, :]
        idx += self.num_tendons
        self.obs_buf[:, idx : idx + self.num_tendons] = self.act_buf[:, :]
        idx += self.num_tendons
        self.obs_buf[:, idx : idx + self.num_tendons] = self.prev_act_buf[:, :]
        idx += self.num_tendons
        self.obs_buf[:, idx : idx + self.num_tendons] = self._angle[:, :]

        print("observations: {}".format(self.obs_buf[0, :]))

    def compute_reward_termination_truncation(self):

        # Compute rewards
        self.rew_buf[:] = 0.0

        # Reward tip position
        tip_pose = self.get_tip_pose_sensor_buf[:, 4:]
        tip_to_goal = self.goal_pose - tip_pose
        tip_to_goal_reward = -1.0 * torch.norm(tip_to_goal, dim=1)
        self.rew_buf[:] += tip_to_goal_reward
        self.info['rewards']["tip_to_goal_reward"] = tip_to_goal_reward.sum().item() / self.total_num_envs

        # Smoothness penalty
        smoothness_penalty = -0.001 * torch.sum((self.act_buf - self.prev_act_buf) ** 2, dim=1)
        self.rew_buf[:] += smoothness_penalty
        self.info['rewards']["smoothness_penalty"] = smoothness_penalty.sum().item() / self.total_num_envs

        # Tendon Vel
        tendon_vel_penalty = -100.0 * torch.sum(self.get_tendon_vel_buf**2, dim=1)
        self.rew_buf[:] += tendon_vel_penalty
        self.info['rewards']["tendon_vel_penalty"] = tendon_vel_penalty.sum().item() / self.total_num_envs

        # Add control penalty
        control_penalty = -0.001 * torch.sum(self.act_buf**2, dim=1)
        self.rew_buf[:] += control_penalty
        self.info['rewards']["control_penalty"] = control_penalty.sum().item() / self.total_num_envs

        # Termination and truncation conditions
        self.term_buf[:] = False # No termination condition for now, we will just use truncation based on episode length
        self.trunc_buf[:] = self.progress_buf >= self.max_episode_length


if __name__ == "__main__":

    from vlearn.utils import get_VL_VISUAL_TESTS

    rendering = True
    with_window = get_VL_VISUAL_TESTS()

    num_iter = np.inf
    max_episode_length = np.inf
    num_envs = 32

    assert torch.cuda.is_available()
    device = torch.device("cuda:0")

    envs = FingerEnvironmentGpu(
        num_envs=num_envs,
        device=device,
        rendering=rendering,
        with_window=with_window,
        max_episode_length=max_episode_length,
    )

    obs, _ = envs.reset()

    gym = v.get_gym()
    render = gym.get_render()

    if render is not None:
        # render.reset_camera(v.Vec3(0.0, 0.2, 0.2), v.Vec3(0.023272, -0.929027, -0.369280))
        render.capped_step = True
        render.set_paused(False)
        reset_box = v.UserCheckbox("Reset", False)
        render.register_menu_item(reset_box)

        sliders = []
        for i, tendon_def in enumerate(envs.art_def.get_spatial_tendon_defs()):
            name = tendon_def.name if tendon_def.name else f"Tendon_{i}"
            sliders.append(v.UserSlider(name, 0, 20, 0.0))
            render.register_menu_item(sliders[-1])

        def control_by_menu():
            if reset_box.get_value():
                envs.reset()
                reset_box.set_value(False)
            actions = torch.tensor([slider.get_value() for slider in sliders], dtype=torch.float32, device=device)

            return torch.tile(actions, (num_envs, 1))

        control_fn = control_by_menu

    done = False
    step = 0
    while not done:
        actions = control_fn()
        # actions = torch.ones(envs.num_tendons, dtype=torch.float32, device=device) * 1.0

        obs, rewards, terminations, truncations, infos = envs.step(actions)

        done = envs.render_finished

        step += 1
        if step >= num_iter:
            done = True

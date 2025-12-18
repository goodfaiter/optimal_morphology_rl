import math
import torch
import numpy as np
from vlearn.spaces import Box
from typing import Tuple, List
import vlearn as v
# from tools.vlearn.train.envs.environment import EnvironmentGpu
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '/workspace/tools/vlearn/train/envs/'))
from environment import EnvironmentGpu
from vlearn.torch_utils.torch_jit_utils import scale, quat_mul, quat_conjugate, v_rpy_from_quat

@torch.jit.script
def reset_noise_helper(init_val: torch.Tensor, noise_scale: float, scale1: float, scale2: float):
    """Generate reset noise for state initialization."""
    return noise_scale * (torch.rand(init_val.shape, dtype=torch.float32,
                          device=init_val.device) * scale1 - scale2)

@torch.jit.script
def compute_reward_termination_truncation_helper(
    actions: torch.Tensor,
    rew_buf: torch.Tensor,
    term_buf: torch.Tensor,
    trunc_buf: torch.Tensor,
    progress_buf: torch.Tensor,
    max_episode_length: int,
    pen_rot: torch.Tensor,
    pen_pos: torch.Tensor,
    pen_goal_rot: torch.Tensor,
):
    """Compute rewards and episode termination conditions."""
    # Maximize pen height
    pen_pos_z = pen_pos[:, -1]
    height_reward = 2.0 * pen_pos_z
    rew_buf[:] = height_reward

    # Reward pen upright orientation
    quat_diff = quat_mul(pen_rot, quat_conjugate(pen_goal_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, :3], p=2, dim=-1), max=1.0))
    rot_rew = 1.0 * torch.exp(-2.0 * rot_dist)
    rew_buf[:] += rot_rew
    
    # Penalize large actions to encourage smooth control
    action_penalty = -0.001 * torch.sum(actions ** 2, dim=-1)
    rew_buf[:] += action_penalty

    # Penalize object drops and end episode
    drop_penalty = pen_pos[:, -1] < -0.1
    rew_buf[:] += -10.0 * drop_penalty
    term_buf[:] = drop_penalty
    trunc_buf[:] = progress_buf >= max_episode_length

class MorphHandPenEnvironmentGpu(EnvironmentGpu):
    """
    Morphological hand environment with with pen interaction.
    
    Features:
    - 16 revolute joints (actuated)
    - 15 morphing joints (shape adaptation)
    - 11 tendons (in tendon control mode)
    - Pen interaction object
    
    Control modes:
    - 'pid': Direct joint position control
    - 'tendon': Cable-driven tendon control
    - Morphing control can be enabled/disabled
    """
    
    # Constants
    NUM_DOFS = 31
    NUM_REVOLUTE = 16
    NUM_MORPH = 15
    NUM_TENDONS = 11
    
    REVOLUTE_IDX = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25, 27, 29]
    MORPH_IDX = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 26, 28, 30]
    
    def __init__(self,
                 num_envs: int,
                 device: torch.device,
                 rendering: bool = False,
                 enable_scene_query: bool = False,
                 max_episode_length: int = 200,
                 gravity: v.Vec3 = v.Vec3(0, 0, -9.81),
                 timestep: float = 0.01667,
                 frame_skip: int = 1,
                 spacing: float = 0.5,
                 initial_is_paused: bool = False,
                 send_interrupt: bool = False,
                 print_hash: bool = False,
                 max_contact_pairs_per_env: int = 64,
                 has_self_collisions: bool = False,
                 force_mass_inertia_computation: bool = True,
                 with_window: bool = True,
                 reset_noise_scale: float = 1.0,
                 relative_control: bool = True,
                 control_mode: str = 'pid',
                 pen_size: str = 'big',
                 control_morph: bool = False,
                 morph_config: list = None):

        assert control_mode in ['pid', 'tendon'], f"Invalid control mode: {control_mode}"
        assert pen_size in ['small', 'mid', 'big'], f"Invalid pen size: {pen_size}"
        
        # Comment this to try out tendon control from random positions
        assert control_mode == 'pid', "Only 'pid' control mode is currently supported, 'tendon' control cannot start from random positions, need fixing."
        
        
        self.num_envs = num_envs
        self.device = device
        self.max_episode_length = max_episode_length
        self.control_mode = control_mode
        self.pen_size = pen_size
        self.control_morph = control_morph
        self.relative_control = relative_control
        self.reset_noise_scale = reset_noise_scale
        self.has_self_collisions = has_self_collisions
        self.force_mass_inertia_computation = force_mass_inertia_computation
        self.morph_config = morph_config if morph_config is not None else []

        # Control parameters
        self._setup_control_parameters()

        # Convert index lists to tensors for efficient indexing
        self.revolute_idx_tensor = torch.tensor(self.REVOLUTE_IDX, device=device, dtype=torch.int)
        self.morph_idx_tensor = torch.tensor(self.MORPH_IDX, device=device, dtype=torch.int)
        
        super().__init__(
            num_envs, device, rendering, enable_scene_query, max_episode_length,
            timestep, frame_skip, spacing, gravity, self.NUM_DOFS, 
            initial_is_paused=initial_is_paused,
            send_interrupt=send_interrupt, 
            up_axis=v.Vec3(0, 0, 1), 
            print_hash=print_hash,
            max_contact_pairs_per_env=max_contact_pairs_per_env, 
            with_window=with_window
        )
        
        print(f"CONTROL MODE: {control_mode}, Morph Control: {control_morph}")
        print("Morph Config Joints:", self.morph_config)

        # Setup action and observation spaces
        self._setup_action_space()
        self._setup_observation_space()
        
        # Create environments
        self.create_envs()

        # Store initial conditions
        self.store_initial_conditions()

        # Allocate buffers
        self.allocate_buffers()

        # Create GPU commands
        self.create_gpu_commands()

        # Finalize gym
        self.gym.gym_finalize()

    def _setup_control_parameters(self):
        """Initialize control-specific parameters."""
        self.force_scale_factor = -480.0  # Tendon force scale
        
        # Relative control scales
        self.tendon_scale = torch.full((self.NUM_TENDONS,), 0.1, device=self.device)
        self.revolute_scale = torch.full((self.NUM_REVOLUTE,), math.pi / 10, device=self.device)
        self.morph_scale = torch.full((self.NUM_MORPH,), 0.0005, device=self.device)

    def _setup_action_space(self):
        """Configure action space dimensions."""
        if self.control_mode == 'tendon':
            self.num_actions = self.NUM_TENDONS
        else:  # pid
            self.num_actions = self.NUM_REVOLUTE
        
        if self.control_morph:
            self.num_actions += self.NUM_MORPH
        
        print(f"Action space size: {self.num_actions}")
        
        self.single_action_space = Box(
            low=np.full(self.num_actions, -1.0, dtype=np.float32),
            high=np.full(self.num_actions, 1.0, dtype=np.float32),
            dtype=np.float32
        )
    
    def _setup_observation_space(self):
        """Configure observation space dimensions."""
        # Joint positions + velocities + ball state + actions
        self.num_obs = 2 * self.NUM_REVOLUTE  # Joint positions + velocities
        self.num_obs += 6 + 6  # Pen pose (euler + pos) + ball vel (linear + angular)
        self.num_obs += 2 * self.num_actions  # Current actions + last actions
        self.num_obs += 3 # Pen goal rotation (euler)

        if self.control_morph:
            self.num_obs += 2*self.NUM_MORPH # Morph positions and velocities
        
        print(f"Observation space size: {self.num_obs}")
        
        self.single_observation_space = Box(
            low=np.full(self.num_obs, np.finfo('f').min, dtype=np.float32),
            high=np.full(self.num_obs, np.finfo('f').max, dtype=np.float32),
            dtype=np.float32
        )

    def create_envs(self):
        """Create simulation environments."""
        # Create environment definition
        self.env_def_handle = self.gym.create_environment_def("hand_env")
        env_def = self.gym.get_environment_def(self.env_def_handle)
        
        # Load appropriate hand model
        hand_files = {
            'tendon': "/workspace/assets/hands/morph_tendon_hand.vsim",
            'pid': "/workspace/assets/hands/morph_pid_hand.vsim"
        }
        
        env_def.import_definitions(
            hand_files[self.control_mode],
            fixed=True,
            use_visual_mesh=False,
            force_mass_computation=self.force_mass_inertia_computation,
            force_inertia_computation=self.force_mass_inertia_computation
        )
        
        # Configure articulation
        self.def_handle = env_def.get_articulation_def_handle_by_name("hand")
        self.art_def = env_def.get_articulation_def(self.def_handle)
        self.art_def.has_self_collisions = self.has_self_collisions
        
        if self.has_self_collisions:
            self.art_def.contact_offset = 0.005
        
        # Set initial pose
        self.root_trans_init = v.Transform(
            v.Quat(0, 0, 0, 1),
            v.Vec3(0, 0, 0)
        )
        self.root_vel_init = v.SpatialVector(0)
        
        # Create articulation
        self.arti_handle = env_def.create_articulation(
            self.def_handle,
            self.root_trans_init,
            'hand'
        )
        
        # Validate dimensions
        assert self.NUM_DOFS == self.art_def.get_num_joint_dof_defs()
        if self.control_mode == 'tendon':
            assert self.NUM_TENDONS == self.art_def.get_num_spatial_tendon_defs()
        
        # Configure PID gains
        self._configure_pid_gains()

        # Load and configure pen
        self._create_pen(env_def)
        
        env_def.finalize()
        super().create_envs(self.env_def_handle)

    def _configure_pid_gains(self):
        """Set PID controller gains for joints."""
        for i in range(self.art_def.get_num_pid_defs()):
            pid_data = self.art_def.get_pid_def(i)
            
            if pid_data.name and pid_data.name.endswith("morph"):
                # High stiffness for morphing joints
                pid_data.stiffness = 1e8
                pid_data.damping = 0.0
            else:
                # Normal gains for revolute joints
                pid_data.stiffness = 3.0
                pid_data.damping = 0.7
                pid_data.max_force = 10.0

    def _create_pen(self, env_def):
        """Load and configure the pen object."""
        # Load appropriate pen model
        pen_files = {
            'small': "/workspace/assets/objects/pen_small.vsim",
            'mid': "/workspace/assets/objects/pen_mid.vsim",
            'big': "/workspace/assets/objects/pen_big.vsim"
        }
        print(f"Loading pen of size: {self.pen_size}")
        
        env_def.import_definitions(
            pen_files[self.pen_size],
            fixed=False,
            use_visual_mesh=False,
            force_mass_computation=self.force_mass_inertia_computation,
            force_inertia_computation=self.force_mass_inertia_computation
        )
        
        # Get pen definition
        pen_def_handle = env_def.get_rigid_body_def_handle_by_name("pen")
        
        # Set initial pen pose
        self.pen_root_trans_init = v.Transform(
            v.Quat(0, 0, 0.7819, 0.6234), 
            v.Vec3(0.07, 0, 0.15)
        )
        self.pen_root_vel_init = v.SpatialVector(0)
        
        # Create pen
        self.pen_handle = env_def.create_rigid_body(
            pen_def_handle,
            self.pen_root_trans_init,
            'pen'
        )

    def store_initial_conditions(self):
        """Extract and store joint and tendon limits."""
        dof_pos_low, dof_pos_high, dof_pos_init = [], [], []
        
        # Extract DOF limits
        for i, dofdef in enumerate(self.art_def.get_joint_dof_defs()):
            dof_pos_low.append(dofdef.low_limit)
            dof_pos_high.append(dofdef.high_limit)
            
            # Morph joints start at max extension
            if dofdef.name and dofdef.name.endswith("morph"):
                if i in self.morph_config:
                    init_pos = dofdef.low_limit
                else:   
                    init_pos = dofdef.high_limit
            else:
                if self.control_mode == 'pid':
                    init_pos = dofdef.low_limit + (dofdef.high_limit - dofdef.low_limit) / 2.0
                else:
                    init_pos = 0.0  # Tendon control starts at zero
            
            dof_pos_init.append(np.clip(init_pos, dofdef.low_limit, dofdef.high_limit))
        
        # Convert to tensors
        self.dof_pos_low = torch.tensor(dof_pos_low, dtype=torch.float32, device=self.device)
        self.dof_pos_high = torch.tensor(dof_pos_high, dtype=torch.float32, device=self.device)
        self.dof_pos_init = torch.tensor(dof_pos_init, dtype=torch.float32, device=self.device)
        self.dof_vel_init = torch.zeros_like(self.dof_pos_init)
        
        # Setup action limits based on control mode
        self._setup_action_limits()

    def _setup_action_limits(self):
        """Configure action space limits based on control mode."""
        if self.control_mode == 'tendon':
            # Extract tendon limits
            tendon_limits = [(td.low_limit, td.high_limit) 
                           for td in self.art_def.get_spatial_tendon_defs()]
            
            tendon_pos_low, tendon_pos_high = zip(*tendon_limits)
            self.tendon_pos_low = torch.tensor(tendon_pos_low, dtype=torch.float32, device=self.device)
            self.tendon_pos_high = torch.tensor(tendon_pos_high, dtype=torch.float32, device=self.device)
            self.tendon_pos_init = self.tendon_pos_high.clone()
            
            self.action_low = self.tendon_pos_low
            self.action_high = self.tendon_pos_high
            
            if self.control_morph:
                morph_low = self.dof_pos_low[self.morph_idx_tensor]
                morph_high = self.dof_pos_high[self.morph_idx_tensor]
                self.action_low = torch.cat([self.action_low, morph_low])
                self.action_high = torch.cat([self.action_high, morph_high])
        
        else:  # pid mode
            if self.control_morph:
                self.action_low = self.dof_pos_low
                self.action_high = self.dof_pos_high
            else:
                self.action_low = self.dof_pos_low[self.revolute_idx_tensor]
                self.action_high = self.dof_pos_high[self.revolute_idx_tensor]

   
    def allocate_buffers(self):
        """Allocate GPU buffers for state and control."""
        super().allocate_buffers()

        # Set hand kinematics
        self.gpu_init_dof_pos = self.dof_pos_init.unsqueeze(0).repeat(self.num_envs, 1)
        self.gpu_init_dof_vel = torch.zeros_like(self.gpu_init_dof_pos)

        # Root transform and velocity buffers
        self._init_root_state_buffers()
        
        # Mode-specific buffers
        if self.control_mode == 'tendon':
            self._allocate_tendon_buffers()
        
        # PID target buffer (used in all modes)
        self.set_pid_target_buf = torch.zeros(
            (self.num_envs, self.NUM_DOFS),
            device=self.device,
            dtype=torch.float32
        )

        # Pen state buffers
        self._allocate_pen_buffers()

        # Last action buffer
        self.last_act_buf = torch.zeros_like(self.act_buf)

    def _init_root_state_buffers(self):
        """Initialize root transform and velocity buffers."""
        # Root velocities
        root_vel = torch.tensor([
            self.root_vel_init.top.x, self.root_vel_init.top.y, self.root_vel_init.top.z,
            self.root_vel_init.bottom.x, self.root_vel_init.bottom.y, self.root_vel_init.bottom.z
        ], dtype=torch.float32, device=self.device)
        self.gpu_init_root_velocities = root_vel.unsqueeze(0).repeat(self.num_envs, 1)
        
        # Root transforms
        root_trans = torch.tensor([
            self.root_trans_init.q.x, self.root_trans_init.q.y,
            self.root_trans_init.q.z, self.root_trans_init.q.w,
            self.root_trans_init.p.x, self.root_trans_init.p.y, self.root_trans_init.p.z
        ], dtype=torch.float32, device=self.device)
        self.gpu_init_root_transforms = root_trans.unsqueeze(0).repeat(self.num_envs, 1)
    
    def _allocate_tendon_buffers(self):
        """Allocate buffers specific to tendon control."""
        self.set_tendon_controls_buf = torch.zeros(
            (self.num_envs, self.NUM_TENDONS),
            device=self.device,
            dtype=torch.float32
        )
        
        self.get_tendon_lengths_buf = torch.zeros_like(self.set_tendon_controls_buf)
        self.get_tendon_vel_buf = torch.zeros_like(self.set_tendon_controls_buf)
        self.get_tendon_forces_buf = torch.zeros_like(self.set_tendon_controls_buf)

    def _allocate_pen_buffers(self):
        """Allocate buffers for pen state and control."""
        # Pen state buffers
        self.get_pen_pos_buf = torch.zeros(
            (self.num_envs, 7), device=self.device, dtype=torch.float32
        )
        self.get_pen_vel_buf = torch.zeros(
            (self.num_envs, 6), device=self.device, dtype=torch.float32
        )
        
        # Pen kinematics control buffers
        self.set_pen_pos_buf = torch.zeros(
            (self.num_envs, 7), device=self.device, dtype=torch.float32
        )

        self.set_pen_vel_buf = torch.zeros(
            (self.num_envs, 6), device=self.device, dtype=torch.float32
        )
        
        # Pen initial state
        pen_vel = torch.tensor([
            self.pen_root_vel_init.top.x, self.pen_root_vel_init.top.y,
            self.pen_root_vel_init.top.z, self.pen_root_vel_init.bottom.x,
            self.pen_root_vel_init.bottom.y, self.pen_root_vel_init.bottom.z
        ], dtype=torch.float32, device=self.device)
        self.gpu_init_pen_velocities = pen_vel.unsqueeze(0).repeat(self.num_envs, 1)
        
        pen_trans = torch.tensor([
            self.pen_root_trans_init.q.x, self.pen_root_trans_init.q.y,
            self.pen_root_trans_init.q.z, self.pen_root_trans_init.q.w,
            self.pen_root_trans_init.p.x, self.pen_root_trans_init.p.y,
            self.pen_root_trans_init.p.z
        ], dtype=torch.float32, device=self.device)
        self.gpu_init_pen_transforms = pen_trans.unsqueeze(0).repeat(self.num_envs, 1)
        
        self.set_pen_pos_buf[:] = self.gpu_init_pen_transforms
        self.set_pen_vel_buf[:] = self.gpu_init_pen_velocities

        # Goal pen rotation (upright)
        self.pen_goal_rot = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)
        self.pen_goal_rot[:, 1] = 0.7071
        self.pen_goal_rot[:, 3] = 0.7071 

        self.pen_goal_rot_euler = v_rpy_from_quat(self.pen_goal_rot)
    
    def create_gpu_commands(self):
        """Create GPU command arrays for efficient state queries and control."""
        # Joint state commands
        self._create_joint_state_commands()
        
        # Kinematic state command
        self._create_kinematic_state_command()
        
        # Mode-specific commands
        if self.control_mode == 'tendon':
            self._create_tendon_commands()
        
        # PID command (used in all modes)
        self._create_pid_command()

        # Pen commands
        self._create_pen_commands()

    def _create_joint_state_commands(self):
        """Create commands for querying joint positions and velocities."""
        get_pos_cmd = self.env_group.create_joint_state_command(
            v.wrap_gpu_buffer(self.get_dof_pos_buf), self.arti_handle
        )
        self.gpu_get_joint_positions_command_array = \
            self.gym.create_joint_state_command_gpu_array([get_pos_cmd])
        
        get_vel_cmd = self.env_group.create_joint_state_command(
            v.wrap_gpu_buffer(self.get_dof_vel_buf), self.arti_handle
        )
        self.gpu_get_joint_velocities_command_array = \
            self.gym.create_joint_state_command_gpu_array([get_vel_cmd])

    def _create_kinematic_state_command(self):
        """Create command for setting articulation kinematic state."""
        set_kin_cmd = self.env_group.create_articulation_kinematic_state_command(
            v.wrap_gpu_buffer(self.set_dof_pos_buf),
            v.wrap_gpu_buffer(self.set_dof_vel_buf),
            v.wrap_gpu_buffer(self.gpu_init_root_transforms),
            v.wrap_gpu_buffer(self.gpu_init_root_velocities),
            self.arti_handle,
            (0, self.NUM_DOFS),
            (0, 1),
            masks_buffer=v.wrap_gpu_buffer(self.reset_buf)
        )
        self.gpu_set_kinematic_state_command_array = \
            self.gym.create_articulation_kinematic_state_command_gpu_array([set_kin_cmd])
    
    def _create_tendon_commands(self):
        """Create commands for tendon control and state queries."""
        # Control command
        set_tendon_cmd = self.env_group.create_spatial_tendon_control_command(
            v.wrap_gpu_buffer(self.set_tendon_controls_buf), self.arti_handle
        )
        self.gpu_set_tendon_control_command_array = \
            self.gym.create_spatial_tendon_control_command_gpu_array([set_tendon_cmd])
        
        # State query commands
        get_length_cmd = self.env_group.create_spatial_tendon_state_command(
            v.SpatialTendonState.LENGTH,
            v.wrap_gpu_buffer(self.get_tendon_lengths_buf),
            self.arti_handle,
            (0, self.NUM_TENDONS)
        )
        self.gpu_get_tendon_lengths_command_array = \
            self.gym.create_spatial_tendon_state_command_gpu_array([get_length_cmd])
        
        get_vel_cmd = self.env_group.create_spatial_tendon_state_command(
            v.SpatialTendonState.VELOCITY,
            v.wrap_gpu_buffer(self.get_tendon_vel_buf),
            self.arti_handle,
            (0, self.NUM_TENDONS)
        )
        self.gpu_get_tendon_velocities_command_array = \
            self.gym.create_spatial_tendon_state_command_gpu_array([get_vel_cmd])
        
        get_force_cmd = self.env_group.create_spatial_tendon_state_command(
            v.SpatialTendonState.FORCE,
            v.wrap_gpu_buffer(self.get_tendon_forces_buf),
            self.arti_handle,
            (0, self.NUM_TENDONS)
        )
        self.gpu_get_tendon_forces_command_array = \
            self.gym.create_spatial_tendon_state_command_gpu_array([get_force_cmd])
    
    def _create_pid_command(self):
        """Create command for PID control."""
        set_pid_cmd = self.env_group.create_pid_control_command(
            v.wrap_gpu_buffer(self.set_pid_target_buf),
            self.arti_handle,
            (0, self.NUM_DOFS)
        )
        self.gpu_set_pid_cmd_arr = \
            self.gym.create_pid_control_command_gpu_array([set_pid_cmd])
        
    def _create_pen_commands(self):
        """Create commands for pen state queries and control."""
        # Get pen state
        get_pen_kin_cmd = self.env_group.create_rigid_body_kinematic_state_command(
            v.wrap_gpu_buffer(self.get_pen_pos_buf),
            v.wrap_gpu_buffer(self.get_pen_vel_buf),
            self.pen_handle
        )
        self.gpu_get_pen_kin_cmd_array = \
            self.gym.create_rigid_body_kinematic_state_command_gpu_array([get_pen_kin_cmd])
        
        # Set pen state (for reset)
        set_pen_kin_cmd = self.env_group.create_rigid_body_kinematic_state_command(
            v.wrap_gpu_buffer(self.set_pen_pos_buf),
            v.wrap_gpu_buffer(self.set_pen_vel_buf),
            self.pen_handle,
            masks_buffer=v.wrap_gpu_buffer(self.reset_buf)
        )
        self.gpu_set_pen_kin_cmd_array = \
            self.gym.create_rigid_body_kinematic_state_command_gpu_array([set_pen_kin_cmd])

    def reset_idx(self):
        """Reset environments based on reset_buf mask."""
        # Reset kinematics
        if self.control_mode == 'pid':
            random_pos_init = torch.rand((self.num_envs, self.NUM_DOFS), 
                                    dtype=torch.float32, device=self.device) * \
                        (self.dof_pos_high - self.dof_pos_low) + self.dof_pos_low
            if not self.control_morph:
                random_pos_init[:, self.morph_idx_tensor] = self.gpu_init_dof_pos[:, self.morph_idx_tensor]
            self.set_dof_pos_buf[:] = random_pos_init
        else:
            self.set_dof_pos_buf[:] = self.gpu_init_dof_pos 
        self.set_dof_vel_buf[:] = self.gpu_init_dof_vel
        self.gym.set_articulation_kinematic_states(self.gpu_set_kinematic_state_command_array)
        
        # Reset control buffers
        reset_mask = self.reset_buf.unsqueeze(1)
        
        self.act_buf = torch.where(reset_mask, torch.zeros_like(self.act_buf), self.act_buf)
        self.set_pid_target_buf[:] = torch.where(
            reset_mask.expand(-1, self.NUM_DOFS),
            self.set_dof_pos_buf,
            self.set_pid_target_buf
        )

        if self.control_mode == 'tendon':
            random_pos_init = torch.rand((self.num_envs, self.NUM_TENDONS), 
                        dtype=torch.float32, device=self.device) * \
            (self.tendon_pos_high - self.tendon_pos_low) + self.tendon_pos_low
            self.set_tendon_controls_buf[:] = torch.where(
                reset_mask.expand(-1, self.NUM_TENDONS),
                random_pos_init,
                self.set_tendon_controls_buf
            )

        # Reset pen state with noise 
        self.set_pen_pos_buf[:, 4:] = self.gpu_init_pen_transforms[:,4:] + \
            reset_noise_helper(self.gpu_init_pen_transforms[:, 4:], self.reset_noise_scale, 0.02, 0.005)
        self.set_pen_pos_buf[:, :4] = self.gpu_init_pen_transforms[:,:4] + \
            reset_noise_helper(self.gpu_init_pen_transforms[:, :4], self.reset_noise_scale, 0.8, 0.3)
        self.set_pen_vel_buf[:] = self.gpu_init_pen_velocities
        self.gym.set_rigid_body_kinematic_states(self.gpu_set_pen_kin_cmd_array)

        
        # Reset progress
        self.progress_buf = torch.where(self.reset_buf, torch.zeros_like(self.progress_buf), self.progress_buf)
    
    def reset(self):
        """Reset all environments."""
        self.reset_buf[:] = True
        super().reset()
        self.compute_observations()
        return self.obs_buf.clone(), {}
    
    def pre_physics_step(self, actions: torch.Tensor):
        """Process actions and apply controls."""
        assert isinstance(actions, torch.Tensor)
        
        # Scale actions to control space
        self._scale_actions(actions)
        
        # Apply controls based on mode
        if self.control_mode == 'tendon':
            self._apply_tendon_control()
        else:  # pid
            self._apply_pid_control()
        
        # Clamp and set PID targets
        self.set_pid_target_buf.clamp_(self.dof_pos_low, self.dof_pos_high)
        self.gym.set_joint_target_positions(self.gpu_set_pid_cmd_arr)

    def _scale_actions(self, actions: torch.Tensor):
        """Scale normalized actions to appropriate control ranges."""
        if self.relative_control:
            if self.control_mode == 'tendon':
                if self.control_morph:
                    # Scale tendons and morph separately
                    self.act_buf[:, :self.NUM_TENDONS] = scale(
                        actions[:, :self.NUM_TENDONS],
                        -self.tendon_scale,
                        self.tendon_scale
                    )
                    self.act_buf[:, self.NUM_TENDONS:] = scale(
                        actions[:, self.NUM_TENDONS:],
                        -self.morph_scale,
                        self.morph_scale
                    )
                else:
                    self.act_buf[:] = scale(
                        actions,
                        -self.tendon_scale,
                        self.tendon_scale
                    )
            else:  # pid
                if self.control_morph:
                    # Scale revolute and morph separately
                    self.act_buf[:, self.revolute_idx_tensor] = scale(
                        actions[:, self.revolute_idx_tensor],
                        -self.revolute_scale,
                        self.revolute_scale
                    )
                    self.act_buf[:, self.morph_idx_tensor] = scale(
                        actions[:, self.morph_idx_tensor],
                        -self.morph_scale,
                        self.morph_scale
                    )
                else:
                    self.act_buf[:] = scale(
                        actions,
                        -self.revolute_scale,
                        self.revolute_scale
                    )
        else:
            self.act_buf[:] = scale(actions, self.action_low, self.action_high)
            self.act_buf[:] /= self.force_scale_factor if self.control_mode == 'tendon' else 1.0
            
    def _apply_tendon_control(self):
        """Apply tendon forces."""
        if self.control_morph:
            tendon_actions = self.act_buf[:, :self.NUM_TENDONS]
            morph_actions = self.act_buf[:, self.NUM_TENDONS:]
            
            if self.relative_control:
                self.set_tendon_controls_buf += tendon_actions * self.force_scale_factor
                self.set_pid_target_buf[:, self.morph_idx_tensor] += morph_actions
            else:
                self.set_tendon_controls_buf[:] = tendon_actions
                self.set_pid_target_buf[:, self.morph_idx_tensor] = morph_actions
        else:
            if self.relative_control:
                self.set_tendon_controls_buf += self.act_buf * self.force_scale_factor
            else:
                self.set_tendon_controls_buf[:] = self.act_buf * self.force_scale_factor
            
            # Lock morph joints at initial position
            self.set_pid_target_buf[:, self.morph_idx_tensor] = \
                self.gpu_init_dof_pos[:, self.morph_idx_tensor]
        
        # Clamp and apply tendon forces
        self.set_tendon_controls_buf.clamp_(self.tendon_pos_low, self.tendon_pos_high)
        self.gym.set_spatial_tendon_forces(self.gpu_set_tendon_control_command_array)

    def _apply_pid_control(self):
        """Apply PID position targets."""
        if self.control_morph:
            if self.relative_control:
                self.set_pid_target_buf += self.act_buf
            else:
                self.set_pid_target_buf[:] = self.act_buf
        else:
            if self.relative_control:
                self.set_pid_target_buf[:, self.revolute_idx_tensor] += self.act_buf
            else:
                self.set_pid_target_buf[:, self.revolute_idx_tensor] = self.act_buf
            
            # Lock morph joints at initial position
            self.set_pid_target_buf[:, self.morph_idx_tensor] = \
                self.gpu_init_dof_pos[:, self.morph_idx_tensor]
    

    def refresh_buffers(self):
        """Refresh all state buffers from simulation."""
        self.gym.get_joint_positions(self.gpu_get_joint_positions_command_array)
        self.gym.get_joint_velocities(self.gpu_get_joint_velocities_command_array)
        self.gym.get_rigid_body_kinematic_states(self.gpu_get_pen_kin_cmd_array)

        if self.control_mode == 'tendon':
            self.gym.get_spatial_tendon_states(self.gpu_get_tendon_lengths_command_array)
            self.gym.get_spatial_tendon_states(self.gpu_get_tendon_velocities_command_array)
            self.gym.get_spatial_tendon_states(self.gpu_get_tendon_forces_command_array)

    def post_physics_step(self):
        """Post-step processing."""
        self.progress_buf += 1
        
        # Check for episode termination
        self.reset_buf[:] = torch.logical_or(self.term_buf, self.trunc_buf)
        self.reset_idx()
        
        # Update state
        self.refresh_buffers()
        self.compute_observations()
        self.compute_reward_termination_truncation()
    
    def compute_observations(self):
        """Construct observation vector."""
        pen_rot_euler = v_rpy_from_quat(self.get_pen_pos_buf[:, 0:4])
        if self.control_morph:
            self.obs_buf[:] = torch.cat([
                self.get_dof_pos_buf,
                self.get_dof_vel_buf,
                pen_rot_euler, # Pen rotation in euler
                self.get_pen_pos_buf[:, 3:6], # Pen position
                self.get_pen_vel_buf,
                self.act_buf,
                self.last_act_buf,
                self.pen_goal_rot_euler,
            ], dim=-1)
        else:
            self.obs_buf[:] = torch.cat([
                self.get_dof_pos_buf[:, self.revolute_idx_tensor],
                self.get_dof_vel_buf[:, self.revolute_idx_tensor],
                pen_rot_euler, # Pen rotation in euler
                self.get_pen_pos_buf[:, 3:6], # Pen position 
                self.get_pen_vel_buf,
                self.act_buf,
                self.last_act_buf,
                self.pen_goal_rot_euler,
            ], dim=-1)
    
    def compute_reward_termination_truncation(self):
        """Compute rewards and check for episode termination."""
        compute_reward_termination_truncation_helper(
            self.act_buf,
            self.rew_buf,
            self.term_buf,
            self.trunc_buf,
            self.progress_buf,
            self.max_episode_length,
            self.get_pen_pos_buf[:, 0:4], # Pen rotation
            self.get_pen_pos_buf[:, 4:7], # Pen position
            self.pen_goal_rot,
        )


if __name__ == "__main__":
    from vlearn.utils import get_VL_VISUAL_TESTS
    
    # Configuration
    config = {
        'num_envs': 1,
        'rendering': True,
        'with_window': get_VL_VISUAL_TESTS(),
        'max_episode_length': 1000,
        'relative_control': True,
        'control_mode': 'pid',  # 'tendon' or 'pid'
        'pen_size': 'small',  # 'small', 'mid', or 'big'
        'control_morph': False
    }
    
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda:0")
    
    # Create environment
    envs = MorphHandPenEnvironmentGpu(device=device, **config)
    obs, _ = envs.reset()
    
    gym = v.get_gym()
    render = gym.get_render()
    
    # Setup interactive control
    if render is not None:
        render.reset_camera(
            v.Vec3(0.0, 0.2, 0.2),
            v.Vec3(0.023272, -0.929027, -0.369280)
        )
        render.capped_step = True
        render.set_paused(False)
        
        # UI elements
        reset_box = v.UserCheckbox("Reset", False)
        render.register_menu_item(reset_box)
        
        # Create sliders based on control mode
        sliders = []
        
        if config['control_mode'] == 'tendon':
            # Tendon sliders
            for i, tendon_def in enumerate(envs.art_def.get_spatial_tendon_defs()):
                name = tendon_def.name or f"Tendon_{i}"
                sliders.append(v.UserSlider(name, -1, 1, 0.0))
                render.register_menu_item(sliders[-1])
            
            # Morph sliders if enabled
            if config['control_morph']:
                for i, dof_def in enumerate(envs.art_def.get_joint_dof_defs()):
                    if dof_def.name and dof_def.name.endswith("morph"):
                        name = dof_def.name or f"Morph_{i}"
                        sliders.append(v.UserSlider(name, -1, 1, 0.0))
                        render.register_menu_item(sliders[-1])
        
        else:  # pid mode
            for i, dof_def in enumerate(envs.art_def.get_joint_dof_defs()):
                # Skip morph joints if not controlling them
                if not config['control_morph'] and dof_def.name and dof_def.name.endswith("morph"):
                    continue
                
                name = dof_def.name or f"DOF_{i}"
                sliders.append(v.UserSlider(name, -1, 1, 0.0))
                render.register_menu_item(sliders[-1])
        
        def control_by_menu():
            """Read control inputs from UI."""
            if reset_box.get_value():
                envs.reset()
                reset_box.set_value(False)
            
            actions = torch.tensor(
                [slider.get_value() for slider in sliders],
                dtype=torch.float32,
                device=device
            )
            
            return actions.unsqueeze(0).expand(config['num_envs'], -1)
        
        control_fn = control_by_menu
    
    # Main simulation loop
    print("\nStarting simulation...")
    step = 0
    while not envs.render_finished:
        actions = control_fn()
        obs, rewards, terminations, truncations, infos = envs.step(actions)
        step += 1
    
    print(f"Simulation completed after {step} steps")
import os
import numpy as np
import torch
from typing import Optional

from vlearn import Vec3
from optimal_morphology_rl.envs.hand_envs.hand_simple_env import HandSimpleEnvironment


def map_user_actions_to_env(actions_7: torch.Tensor, env: HandSimpleEnvironment) -> torch.Tensor:
    """
    Map user-provided actions of shape (N,7): [lin_x, lin_y, lin_z, ang_x, ang_y, ang_z, grasp]
    into the environment's action vector shape `(N, env.num_actions)`.

    Mapping rules:
    - env expects root velocities in order [ang(3), lin(3)] in the first 6 entries.
    - remaining entries (6:) are motor/grasp commands; we broadcast the single `grasp` value
      to all motors.
    """
    device = actions_7.device
    N = actions_7.shape[0]
    num_actions = env.num_actions
    out = torch.zeros((N, num_actions), device=device, dtype=torch.float32)

    # User format: [lin(3), ang(3), grasp]
    lin = actions_7[:, 0:3]
    ang = actions_7[:, 3:6]
    grasp = actions_7[:, 6:7]

    # env expects ang first, then lin
    out[:, 0:3] = ang
    out[:, 3:6] = lin

    if num_actions > 6:
        # broadcast grasp to all motor entries (or leave zeros if env defines more sophisticated motor mapping)
        out[:, 6:] = grasp.expand(-1, num_actions - 6)

    return out


def load_actions_or_sample(
    actions_npz_path: Optional[str],
    num_envs: int,
    timesteps: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Load actions from an NPZ file when provided, otherwise sample from a normal distribution.
    Adds a random sinusoid to each action dimension with independent random frequency, amplitude, and phase.

    The loaded data must have shape (timesteps, num_envs, 7) and represent user actions in the
    [lin_x, lin_y, lin_z, ang_x, ang_y, ang_z, grasp] layout.
    """
    if actions_npz_path:
        if not os.path.exists(actions_npz_path):
            raise FileNotFoundError(f"Actions NPZ not found: {actions_npz_path}")

        with np.load(actions_npz_path, allow_pickle=True) as loaded:
            if isinstance(loaded, np.lib.npyio.NpzFile):
                if "actions" in loaded.files:
                    actions = loaded["actions"]
                elif len(loaded.files) == 1:
                    actions = loaded[loaded.files[0]]
                else:
                    raise KeyError(f"Expected an 'actions' array in {actions_npz_path} or a single array, got keys {loaded.files}")
            else:
                actions = loaded

        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 3 or actions.shape[-1] != 7:
            raise ValueError(f"Expected actions shape (timesteps, num_envs, 7), got {actions.shape}")
        if actions.shape[1] != num_envs:
            raise ValueError(f"Expected actions second dimension {num_envs}, got {actions.shape[1]}")

        return actions

    # Add random sinusoids to each action dimension (per environment)
    actions = np.random.normal(0, 1.0, size=(timesteps, num_envs, 7)).astype(np.float32)
    actions[:] = 0
    rng_sin = np.random.default_rng((0 if seed is None else seed) + 12345)

    # Define amplitude per action dimension: [lin_x, lin_y, lin_z, ang_x, ang_y, ang_z, grasp]
    amplitudes = np.array([3.0, 3.0, 3.0, 5.0, 5.0, 5.0, 1.0], dtype=np.float32)

    time = np.arange(timesteps, dtype=np.float32)
    for dim in range(7):
        # Generate random frequency and phase for each environment
        frequencies = rng_sin.uniform(0.005, 0.1, size=num_envs)  # (num_envs,)
        phases = rng_sin.uniform(0, 2 * np.pi, size=num_envs)      # (num_envs,)
        
        # Compute sinusoid for all timesteps and all environments
        sinusoid = amplitudes[dim] * np.sin(2 * np.pi * frequencies[np.newaxis, :] * time[:, np.newaxis] + phases[np.newaxis, :])
        actions[:, :, dim] = sinusoid

    return actions


def load_initial_conditions_or_sample(
    initial_conditions_npz_path: Optional[str],
    num_envs: int,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if initial_conditions_npz_path:
        if not os.path.exists(initial_conditions_npz_path):
            raise FileNotFoundError(f"Initial conditions NPZ not found: {initial_conditions_npz_path}")

        with np.load(initial_conditions_npz_path, allow_pickle=True) as loaded:
            if "hand_reset_pos_buf" not in loaded or "hand_reset_quat_buf" not in loaded:
                raise KeyError(
                    f"Expected hand_reset_pos_buf and hand_reset_quat_buf in {initial_conditions_npz_path}, got keys {loaded.files}"
                )

            hand_reset_pos_buf = np.asarray(loaded["hand_reset_pos_buf"], dtype=np.float32)
            hand_reset_quat_buf = np.asarray(loaded["hand_reset_quat_buf"], dtype=np.float32)

        if hand_reset_pos_buf.shape != (num_envs, 3):
            raise ValueError(f"Expected hand_reset_pos_buf shape {(num_envs, 3)}, got {hand_reset_pos_buf.shape}")
        if hand_reset_quat_buf.shape != (num_envs, 4):
            raise ValueError(f"Expected hand_reset_quat_buf shape {(num_envs, 4)}, got {hand_reset_quat_buf.shape}")

        hand_reset_pos = torch.from_numpy(hand_reset_pos_buf)
        hand_reset_quat = torch.from_numpy(hand_reset_quat_buf)
        hand_reset_quat = torch.nn.functional.normalize(hand_reset_quat, dim=-1)
        return hand_reset_pos, hand_reset_quat

    rng = np.random.default_rng(seed)
    hand_reset_pos_buf = np.empty((num_envs, 3), dtype=np.float32)
    hand_reset_pos_buf[:, :] = rng.uniform(-0.5, 0.5, size=(num_envs, 3)).astype(np.float32)\

    hand_reset_quat_buf = rng.normal(0.0, 1.0, size=(num_envs, 4)).astype(np.float32)
    norms = np.linalg.norm(hand_reset_quat_buf, axis=-1, keepdims=True)
    hand_reset_quat_buf = hand_reset_quat_buf / np.clip(norms, 1e-8, None)

    return torch.from_numpy(hand_reset_pos_buf), torch.from_numpy(hand_reset_quat_buf)


def run_recording(
    actions_np: np.ndarray,
    output_path: str,
    hand_vsim: str,
    num_envs: int,
    device_str: str,
    hand_reset_pos_buf: torch.Tensor,
    hand_reset_quat_buf: torch.Tensor,
    visualize: bool = False,
    max_steps: Optional[int] = None,
):
    device = torch.device(device_str)

    max_steps = actions_np.shape[0]
    assert actions_np.shape[1] == num_envs, "Second dim must match num_envs"

    env = HandSimpleEnvironment(
        num_envs=num_envs,
        device=device,
        rendering=visualize,
        with_window=visualize,
        max_episode_length=100000,
        hand_vsim=hand_vsim,
    )
    env.hand_reset_pos_buf = hand_reset_pos_buf.to(device=device, dtype=torch.float32)
    env.hand_reset_quat_buf = hand_reset_quat_buf.to(device=device, dtype=torch.float32)

    obs, _ = env.reset()

    # Set camera for visualization
    if visualize:
        camera_eye = Vec3(-0.814822, 0.720006, 2.227933)
        camera_dir = Vec3(0.681413, -0.554998, -0.477131)
        env.gym.get_render().reset_camera(camera_eye, camera_dir)

    # Prepare storage
    obs_shape = obs.shape  # (num_envs, num_obs)
    recordings = {
        "observations": np.zeros((max_steps, obs_shape[0], obs_shape[1]), dtype=np.float32),
        "actions": np.zeros((max_steps, obs_shape[0], 7), dtype=np.float32),
        "hand_reset_pos_buf": env.hand_reset_pos_buf.detach().cpu().numpy(),
        "hand_reset_quat_buf": env.hand_reset_quat_buf.detach().cpu().numpy(),
    }

    # Run loop
    for t in range(max_steps):
        user_actions = torch.tensor(actions_np[t], dtype=torch.float32, device=device)
        env_actions = map_user_actions_to_env(user_actions, env)

        obs_step, _, _, _, _ = env.step(env_actions)

        recordings["observations"][t] = obs_step.cpu().numpy()
        recordings["actions"][t] = user_actions.cpu().numpy()

    # insert observation dimension into filename before saving
    base, ext = os.path.splitext(output_path)
    output_path_with_obs = f"{base}_obs{obs_shape[1]}{ext}"

    recordings["num_joints"] = env.num_joints
    recordings["num_envs"] = num_envs

    # Save as .npy when requested, otherwise fall back to compressed npz.
    np.savez_compressed(output_path_with_obs, **recordings)
    print(f"Saved recording to {output_path_with_obs}")

def map_recording_to_new_dim(trajectory_npz: str, new_obs_dim: int):
    with np.load(trajectory_npz) as loaded:
        if "observations" not in loaded or "actions" not in loaded:
            raise KeyError(f"Expected 'observations' and 'actions' in {trajectory_npz}, got keys {loaded.files}")
        observations = loaded["observations"]
        old_num_joints = loaded["num_joints"]

    base_num_obs = 15

    num_timesteps = observations.shape[0]
    num_envs = observations.shape[1]
    new_observations = np.zeros((num_timesteps, num_envs, new_obs_dim), dtype=np.float32)
    num_joints = (new_obs_dim - base_num_obs) // 2
    new_observations[:, :, :base_num_obs] = observations[:, :, :base_num_obs]
    average_grasp = np.mean(observations[:, :, base_num_obs:base_num_obs + old_num_joints], axis=2, keepdims=True)
    new_observations[:, :, base_num_obs:base_num_obs + num_joints] = average_grasp[:, :]
    average_vel = np.mean(observations[:, :, base_num_obs + old_num_joints:base_num_obs + 2 * old_num_joints], axis=2, keepdims=True)
    new_observations[:, :, base_num_obs + num_joints:base_num_obs + 2 * num_joints] = average_vel[:, :]

    # Save the new trajectory
    base, ext = os.path.splitext(trajectory_npz)
    output_path = f"{base}_obs{new_obs_dim}{ext}"
    np.savez_compressed(output_path, observations=new_observations)
    print(f"Saved mapped trajectory to {output_path}")


if __name__ == "__main__":
    NUM_ENVS = 4096
    TIMESTEPS = 500
    DEVICE = "cuda:0"
    # HAND_VSIM = "/workspace/optimal_morphology_rl/data/llm/cad_parameters_a_hand_can_pick_up_a_berry.vsim"
    # ACTIONS_NPZ = None
    HAND_VSIM = "/workspace/optimal_morphology_rl/data/llm/iteration_02/variant_2_upgrade_2_hand.vsim"
    ACTIONS_NPZ = "/workspace/data/transfer/training_data/hand_simple_recording_obs41.npz"
    SEED = 200964
    OUTPUT_PATH = "/workspace/data/transfer/training_data/hand_simple_recording"
    VISUALIZE = False

    if ACTIONS_NPZ is not None:
        map_recording_to_new_dim(ACTIONS_NPZ, new_obs_dim=29)
    else:
        actions = load_actions_or_sample(ACTIONS_NPZ, NUM_ENVS, TIMESTEPS, seed=SEED)
        hand_reset_pos_buf, hand_reset_quat_buf = load_initial_conditions_or_sample(ACTIONS_NPZ, NUM_ENVS, seed=SEED)
        run_recording(
            actions,
            OUTPUT_PATH,
            HAND_VSIM,
            NUM_ENVS,
            DEVICE,
            hand_reset_pos_buf=hand_reset_pos_buf,
            hand_reset_quat_buf=hand_reset_quat_buf,
            visualize=VISUALIZE,
            max_steps=TIMESTEPS,
        )

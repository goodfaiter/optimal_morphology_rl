"""Pytest suite for the hand-object environment with a drawer."""

import contextlib
import io
import math
import os

import pytest
import torch
import vlearn as v

from optimal_morphology_rl.envs.hand_envs.hand_object_env import HandObjectEnvironmentGpu


VSIM_PATH = "/workspace/data/vsim/claw_3_tendon.vsim"


def _make_env(max_episode_length: int = 10):
    """Create a single-env drawer environment, suppressing noisy setup output."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    if not os.path.isfile(VSIM_PATH):
        pytest.fail(f"Hand VSIM not found: {VSIM_PATH}")

    device = torch.device("cuda:0")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        env = HandObjectEnvironmentGpu(
            num_envs=1,
            device=device,
            rendering=False,
            with_window=False,
            max_episode_length=max_episode_length,
            vsim_path=VSIM_PATH,
            object="drawer",
        )
    return env


@pytest.fixture(scope="module")
def _env_session():
    """Single vlearn Gym instance shared across tests in this module."""
    env = _make_env()
    yield env
    # Break references held by the env so the singleton Gym can be released
    # before the next test module creates its own gym.
    env.gym = None
    env.env_group = None
    env.env_def_handle = None
    env.robot = None
    env.objects = None
    env.reward_object = None
    env.camera = None
    env.kinematic_sensor = None
    env.forces = None
    env.contacts = None
    v.delete_gym()
    del env


@pytest.fixture
def env(_env_session):
    """Reset the shared environment and drawer joints before each test."""
    drawer = _env_session.objects.get_object("drawer")
    drawer.set_joint_pos_buf.zero_()
    drawer.set_joint_vel_buf.zero_()
    _env_session.reset_buf[:] = True
    _env_session.gym.set_articulation_kinematic_states(drawer.gpu_set_object_kin_cmd_array)
    _env_session.reset_buf[:] = False
    _env_session.reset()
    yield _env_session


class TestHandObjectEnvironmentDrawer:
    """Smoke and behavior tests for the drawer task."""

    def test_launch_and_step_drawer(self, env):
        """The environment resets and steps without crashing."""
        obs, _ = env.reset()
        assert obs.shape[0] == 1

        actions = torch.zeros((1, env.num_actions), device=env.device, dtype=torch.float32)
        obs, reward, terminated, truncated, info = env.step(actions)

        assert obs.shape[0] == 1
        assert reward.shape[0] == 1
        assert terminated.shape[0] == 1
        assert truncated.shape[0] == 1

    def test_drawer_motor_and_goal_state(self, env):
        """The drawer articulation exposes both motors and a 90-degree goal orientation."""
        drawer = env.objects.get_object("drawer")

        assert drawer.spring_motor_cmd_buf is not None
        assert drawer.num_motors == 2
        assert drawer.spring_motor_cmd_buf.shape == (1, drawer.num_motors)

        assert drawer.handle_joint_motor_index is not None
        assert drawer.handle_joint_dof_index is not None
        assert drawer.drawer_joint_motor_index is not None
        assert drawer.drawer_joint_dof_index is not None

        env.reset()
        expected_quat = torch.tensor(
            [math.sin(math.pi / 4.0), 0.0, 0.0, math.cos(math.pi / 4.0)],
            device=env.device,
            dtype=torch.float32,
        )
        assert torch.allclose(drawer.goal_quat_object_to_world[0], expected_quat)

    def test_drawer_stays_locked_with_zero_actions(self, env):
        """With no hand actuation, the drawer prismatic joint stays near zero and locked."""
        drawer = env.objects.get_object("drawer")
        env.reset()

        actions = torch.zeros((1, env.num_actions), device=env.device, dtype=torch.float32)
        for _ in range(10):
            env.step(actions)

        assert not drawer.unlocked_buf[0].item()
        drawer_pos = drawer.get_joint_pos_buf[0, drawer.drawer_joint_dof_index].item()
        assert abs(drawer_pos) < 1e-3

    def test_drawer_unlocks_at_ninety_degrees(self, env):
        """When the handle is rotated past the threshold, the drawer unlocks."""
        drawer = env.objects.get_object("drawer")
        env.reset()

        # Set the handle joint past the unlock threshold (80 degrees).
        target_handle_angle = 1.5  # radians, > 80 deg
        drawer.set_joint_pos_buf[0, drawer.handle_joint_dof_index] = target_handle_angle
        env.reset_buf[0] = True
        env.gym.set_articulation_kinematic_states(drawer.gpu_set_object_kin_cmd_array)
        env.reset_buf[0] = False

        actions = torch.zeros((1, env.num_actions), device=env.device, dtype=torch.float32)

        # First step: physics propagates the set joint position.
        env.step(actions)
        assert drawer.get_joint_pos_buf[0, drawer.handle_joint_dof_index].item() > 1.0

        # Second step: pre_physics_step sees the large handle angle and unlocks.
        env.step(actions)
        assert drawer.unlocked_buf[0].item()
        # Lock force on the drawer motor should now be zero.
        assert drawer.spring_motor_cmd_buf[0, drawer.drawer_joint_motor_index].item() == 0.0

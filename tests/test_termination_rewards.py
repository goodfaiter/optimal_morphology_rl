"""Tests for termination modules that also modify rewards."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.envs.hand_cube.terminations.cube_success_termination import (
    CubeSuccessTermination,
)
from optimal_morphology_rl.modules.terminations.bounds_termination import (
    BoundsTermination,
)
from optimal_morphology_rl.modules.terminations.drop_termination import DropTermination


class _FakeKinematicSensor:
    def __init__(self, pos_in_world: torch.Tensor):
        self.pos_in_world = pos_in_world


class _FakeModuleManager:
    def __init__(self, container: ModuleContainer):
        self.container = container


class _FakeEnv:
    def __init__(self, container: ModuleContainer):
        self.module_manager = _FakeModuleManager(container)
        self.device = container.device
        self.total_num_envs = container.total_num_envs
        self.rew_buf = torch.zeros(
            container.total_num_envs, dtype=torch.float32, device=container.device
        )
        self.term_buf = torch.zeros(
            container.total_num_envs, dtype=torch.bool, device=container.device
        )
        self.trunc_buf = torch.zeros(
            container.total_num_envs, dtype=torch.bool, device=container.device
        )
        self.info: dict[str, Any] = {"rewards": {}}


@pytest.fixture
def drop_setup() -> tuple[_FakeEnv, DropTermination]:
    device = torch.device("cpu")
    total_num_envs = 4
    container = ModuleContainer()
    container.device = device
    container.total_num_envs = total_num_envs
    container.kinematic_sensor = _FakeKinematicSensor(
        torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, -0.2], [0.0, 0.0, 0.1], [0.0, 0.0, -0.15]],
            dtype=torch.float32,
            device=device,
        )
    )

    env = _FakeEnv(container)
    module = DropTermination({"threshold": -0.1, "reward_scale": -20.0})
    return env, module


def test_drop_termination_sets_term_buf(drop_setup) -> None:
    env, module = drop_setup
    module.compute(env)

    assert env.term_buf.tolist() == [False, True, False, True]


def test_drop_termination_adds_negative_reward(drop_setup) -> None:
    env, module = drop_setup
    module.compute(env)

    expected = torch.tensor([0.0, -20.0, 0.0, -20.0], dtype=torch.float32)
    assert torch.allclose(env.rew_buf, expected)
    assert env.info["rewards"]["drop_penalty"] == pytest.approx(-0.5)


def test_drop_termination_zero_scale_disables_reward() -> None:
    device = torch.device("cpu")
    total_num_envs = 2
    container = ModuleContainer()
    container.device = device
    container.total_num_envs = total_num_envs
    container.kinematic_sensor = _FakeKinematicSensor(
        torch.tensor([[0.0, 0.0, -0.2], [0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    )
    env = _FakeEnv(container)
    module = DropTermination({"threshold": -0.1, "reward_scale": 0.0})

    module.compute(env)

    assert env.term_buf.tolist() == [True, False]
    assert torch.all(env.rew_buf == 0.0)


class _FakeTable:
    class _HalfSize:
        def __init__(self, x: float, y: float):
            self.x = x
            self.y = y

    def __init__(self, x: float, y: float):
        self.half_size = self._HalfSize(x, y)


class _FakeRobot:
    def __init__(self, robot_pos_in_world: torch.Tensor):
        self.robot_pos_in_world = robot_pos_in_world


@pytest.fixture
def bounds_setup() -> tuple[_FakeEnv, BoundsTermination]:
    device = torch.device("cpu")
    total_num_envs = 4
    container = ModuleContainer()
    container.device = device
    container.total_num_envs = total_num_envs
    container.objects = {"table": _FakeTable(0.5, 0.3)}
    container.robot = _FakeRobot(
        torch.tensor(
            [
                [0.0, 0.0, 0.1],  # in bounds
                [1.0, 0.0, 0.1],  # out of x bounds
                [0.0, 0.6, 0.1],  # out of y bounds
                [0.0, 0.0, 0.5],  # out of z bounds
            ],
            dtype=torch.float32,
            device=device,
        )
    )

    env = _FakeEnv(container)
    module = BoundsTermination({"padding": 0.2, "reward_scale": -10.0})
    module.post_finalize(container)
    return env, module


def test_bounds_termination_sets_term_buf(bounds_setup) -> None:
    env, module = bounds_setup
    module.compute(env)

    assert env.term_buf.tolist() == [False, True, True, True]


def test_bounds_termination_adds_negative_reward(bounds_setup) -> None:
    env, module = bounds_setup
    module.compute(env)

    expected = torch.tensor([0.0, -10.0, -10.0, -10.0], dtype=torch.float32)
    assert torch.allclose(env.rew_buf, expected)
    assert env.info["rewards"]["bounds_penalty"] == pytest.approx(-0.75)


def test_bounds_termination_zero_scale_disables_reward() -> None:
    device = torch.device("cpu")
    total_num_envs = 2
    container = ModuleContainer()
    container.device = device
    container.total_num_envs = total_num_envs
    container.objects = {"table": _FakeTable(0.5, 0.3)}
    container.robot = _FakeRobot(
        torch.tensor([[1.0, 0.0, 0.1], [0.0, 0.0, 0.1]], dtype=torch.float32, device=device)
    )
    env = _FakeEnv(container)
    module = BoundsTermination({"padding": 0.2, "reward_scale": 0.0})
    module.post_finalize(container)

    module.compute(env)

    assert env.term_buf.tolist() == [True, False]
    assert torch.all(env.rew_buf == 0.0)


class _FakeRewardObject:
    def __init__(self, goal_quat: torch.Tensor):
        self.goal_quat_object_to_world = goal_quat


class _FakeKinematicSensorQuat:
    def __init__(self, quat_sensor_to_world: torch.Tensor):
        self.quat_sensor_to_world = quat_sensor_to_world


def _make_cube_env(
    object_quat: torch.Tensor, goal_quat: torch.Tensor
) -> tuple[_FakeEnv, CubeSuccessTermination]:
    device = object_quat.device
    total_num_envs = object_quat.shape[0]
    container = ModuleContainer()
    container.device = device
    container.total_num_envs = total_num_envs
    container.reward_object_name = "cube"
    container.reward_object = _FakeRewardObject(goal_quat)
    container.kinematic_sensor = _FakeKinematicSensorQuat(object_quat)

    env = _FakeEnv(container)
    module = CubeSuccessTermination({"success_scale": 200.0})
    module.post_finalize(container)
    return env, module


def test_cube_success_termination_adds_success_bonus() -> None:
    device = torch.device("cpu")
    total_num_envs = 2
    # Aligned quaternions.
    goal_quat = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0]] * total_num_envs, dtype=torch.float32, device=device
    )
    object_quat = goal_quat.clone()

    env, module = _make_cube_env(object_quat, goal_quat)

    # 31 aligned steps triggers success.
    for _ in range(30):
        module.compute(env)
        env.rew_buf[:] = 0.0

    assert not env.term_buf.any()

    module.compute(env)
    assert env.term_buf.all()
    expected = torch.full((total_num_envs,), 200.0, dtype=torch.float32)
    assert torch.allclose(env.rew_buf, expected)


def test_cube_success_resets_on_misalignment() -> None:
    device = torch.device("cpu")
    total_num_envs = 1
    goal_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
    # 90 degree rotation around x-axis.
    angle = math.radians(90.0)
    misaligned_quat = torch.tensor(
        [[math.sin(angle / 2), 0.0, 0.0, math.cos(angle / 2)]],
        dtype=torch.float32,
        device=device,
    )

    env, module = _make_cube_env(goal_quat, goal_quat)
    # Run 30 aligned steps.
    for _ in range(30):
        module.compute(env)
        env.rew_buf[:] = 0.0

    # One misaligned step resets the counter.
    env.module_manager.container.kinematic_sensor.quat_sensor_to_world = misaligned_quat
    module.compute(env)
    env.rew_buf[:] = 0.0
    assert not env.term_buf.any()

    # Another aligned step should not yet succeed.
    env.module_manager.container.kinematic_sensor.quat_sensor_to_world = goal_quat
    module.compute(env)
    assert not env.term_buf.any()


def test_cube_success_reports_goal_success_rate() -> None:
    device = torch.device("cpu")
    total_num_envs = 2
    goal_quat = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0]] * total_num_envs, dtype=torch.float32, device=device
    )
    object_quat = goal_quat.clone()

    env, module = _make_cube_env(object_quat, goal_quat)

    # Run 30 aligned steps (not enough to trigger success).
    for _ in range(30):
        module.compute(env)
        env.rew_buf[:] = 0.0
        assert "goal_success_rate" not in env.info["rewards"]

    # 31st step triggers success. Force truncation so episodes end.
    env.trunc_buf[:] = True
    module.compute(env)

    assert "goal_success_rate" in env.info["rewards"]
    # Both envs succeeded and ended, so rate is 1.0.
    assert env.info["rewards"]["goal_success_rate"] == pytest.approx(1.0)

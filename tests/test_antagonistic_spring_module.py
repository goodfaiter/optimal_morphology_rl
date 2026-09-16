"""Unit tests for the antagonistic_spring module."""

import pytest
import torch

from optimal_morphology_rl.modules.antagonistic_spring_module import AntagonisticSpring
from optimal_morphology_rl.modules.module_container import ModuleContainer


class _FakeEnv:
    pass


class _FakeRobot:
    def __init__(self, num_motors: int = 2):
        self.use_tendon = False
        self.num_motors = num_motors
        self.motor_to_joint_dof_index = torch.arange(num_motors, dtype=torch.long)
        self.get_joint_pos_buf = torch.zeros((1, num_motors), dtype=torch.float32)


@pytest.fixture
def container() -> ModuleContainer:
    cont = ModuleContainer()
    cont.total_num_envs = 1
    cont.device = torch.device("cpu")
    cont.env = _FakeEnv()
    cont.robot = _FakeRobot()
    cont.reset_buf = torch.zeros(1, dtype=torch.bool)
    cont.robot.get_joint_pos_buf = torch.tensor([[1.0, -2.0]])
    return cont


def _make_module(spring_constants=0.1) -> AntagonisticSpring:
    return AntagonisticSpring({"spring_constants": spring_constants})


def test_post_finalize_builds_constants_and_allocates_buffer(container: ModuleContainer) -> None:
    module = _make_module(0.2)
    module.finalize(container)
    module.post_finalize(container)

    assert torch.allclose(module.spring_constants, torch.full((2,), 0.2))
    assert container.antagonistic_spring_force_buf.shape == (1, 2)
    assert torch.all(container.antagonistic_spring_force_buf == 0.0)


def test_step_computes_resisting_spring_forces(container: ModuleContainer) -> None:
    module = _make_module(0.1)
    module.finalize(container)
    module.post_finalize(container)
    module.step(container)

    # force = -spring_constants * joint_pos = -0.1 * [1.0, -2.0].
    assert container.antagonistic_spring_force_buf[:, 0].item() == pytest.approx(-0.1)
    assert container.antagonistic_spring_force_buf[:, 1].item() == pytest.approx(0.2)


def test_step_supports_per_motor_constants(container: ModuleContainer) -> None:
    module = _make_module([2.0, 3.0])
    module.finalize(container)
    module.post_finalize(container)
    module.step(container)

    assert container.antagonistic_spring_force_buf[:, 0].item() == pytest.approx(-2.0)
    assert container.antagonistic_spring_force_buf[:, 1].item() == pytest.approx(-3.0 * -2.0)


def test_step_overwrites_previous_buffer_contents(container: ModuleContainer) -> None:
    module = _make_module(0.1)
    module.finalize(container)
    module.post_finalize(container)

    container.antagonistic_spring_force_buf.fill_(5.0)
    container.robot.get_joint_pos_buf = torch.tensor([[1.0, -2.0]])
    module.step(container)

    assert container.antagonistic_spring_force_buf[:, 0].item() == pytest.approx(-0.1)
    assert container.antagonistic_spring_force_buf[:, 1].item() == pytest.approx(0.2)


def test_supports_tendon_robots() -> None:
    cont = ModuleContainer()
    cont.total_num_envs = 1
    cont.device = torch.device("cpu")
    cont.env = _FakeEnv()
    cont.robot = _FakeRobot()
    cont.robot.use_tendon = True
    cont.robot.num_tendons = 3
    cont.robot.get_joint_pos_buf = torch.zeros((1, 2))
    cont.reset_buf = torch.zeros(1, dtype=torch.bool)

    module = _make_module(0.1)
    module.finalize(cont)
    module.post_finalize(cont)

    assert cont.antagonistic_spring_force_buf.shape == (1, 2)


def test_finalize_requires_robot(container: ModuleContainer) -> None:
    container.robot = None
    module = _make_module()
    with pytest.raises(RuntimeError, match="requires 'robot'"):
        module.finalize(container)


def test_finalize_requires_env(container: ModuleContainer) -> None:
    container.env = None
    module = _make_module()
    with pytest.raises(RuntimeError, match="requires 'env'"):
        module.finalize(container)


def test_post_finalize_requires_spring_constants_config(container: ModuleContainer) -> None:
    module = AntagonisticSpring({})
    module.finalize(container)
    with pytest.raises(RuntimeError, match="spring_constants"):
        module.post_finalize(container)


def test_post_finalize_rejects_per_motor_length_mismatch(container: ModuleContainer) -> None:
    module = _make_module([1.0, 2.0, 3.0])  # two motors, three values
    module.finalize(container)
    with pytest.raises(RuntimeError, match="spring_constants"):
        module.post_finalize(container)

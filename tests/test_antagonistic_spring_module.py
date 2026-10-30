"""Unit tests for the antagonistic_spring module."""

from types import SimpleNamespace

import pytest
import torch

from optimal_morphology_rl.modules.antagonistic_spring_module import AntagonisticSpring
from optimal_morphology_rl.modules.module_container import ModuleContainer


class _FakeEnv:
    pass


class _FakeArtDef:
    def get_motor_def(self, i: int):
        return SimpleNamespace(name=f"motor_{i}")


class _FakeRobot:
    def __init__(self, num_motors: int = 2):
        self.use_tendon = False
        self.num_motors = num_motors
        self.art_def = _FakeArtDef()
        self.motor_to_joint_dof_index = torch.arange(num_motors, dtype=torch.long)
        self.revolute_dof_mask = torch.ones(num_motors, dtype=torch.bool)
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


def _make_envs(num_envs: int, joint_pos: torch.Tensor | None = None) -> ModuleContainer:
    cont = ModuleContainer()
    cont.total_num_envs = num_envs
    cont.device = torch.device("cpu")
    cont.env = _FakeEnv()
    cont.robot = _FakeRobot()
    cont.reset_buf = torch.zeros(num_envs, dtype=torch.bool)
    cont.robot.get_joint_pos_buf = (
        joint_pos if joint_pos is not None else torch.zeros((num_envs, 2), dtype=torch.float32)
    )
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


def test_step_skips_non_revolute_motors(container: ModuleContainer) -> None:
    # Motor 1 drives a prismatic dof (a spool slider): no spring on it.
    container.robot.revolute_dof_mask = torch.tensor([True, False])
    module = _make_module(0.1)
    module.finalize(container)
    module.post_finalize(container)
    module.step(container)

    assert container.antagonistic_spring_force_buf[:, 0].item() == pytest.approx(-0.1)
    assert container.antagonistic_spring_force_buf[:, 1].item() == 0.0


def test_per_revolute_motor_constants(container: ModuleContainer) -> None:
    # Three motors: two revolute (springs 0.1 and 1.0), one prismatic slider in between (no spring).
    container.robot = _FakeRobot(num_motors=3)
    container.robot.revolute_dof_mask = torch.tensor([True, False, True])
    container.robot.get_joint_pos_buf = torch.tensor([[1.0, 5.0, -2.0]])
    module = _make_module([0.1, 1.0])
    module.finalize(container)
    module.post_finalize(container)
    module.step(container)

    assert torch.allclose(module.spring_constants, torch.tensor([0.1, 0.0, 1.0]))
    assert torch.allclose(container.antagonistic_spring_force_buf, torch.tensor([[-0.1, 0.0, 2.0]]))


def test_spring_constants_with_wrong_length_raise(container: ModuleContainer) -> None:
    container.robot = _FakeRobot(num_motors=3)
    container.robot.revolute_dof_mask = torch.tensor([True, False, True])
    module = _make_module([0.1, 1.0, 1.0, 1.0])
    module.finalize(container)
    with pytest.raises(RuntimeError, match="length"):
        module.post_finalize(container)


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


def test_post_finalize_fills_the_per_env_buffer_with_the_constants() -> None:
    num_envs = 4
    cont = _make_envs(num_envs, joint_pos=torch.tensor([[1.0, -2.0]] * num_envs))
    module = _make_module([0.1, 0.3])
    module.finalize(cont)
    module.post_finalize(cont)
    module.step(cont)

    assert torch.allclose(module.spring_constants_buf, torch.tensor([[0.1, 0.3]] * num_envs))
    assert torch.allclose(cont.antagonistic_spring_force_buf, torch.tensor([[-0.1, 0.6]] * num_envs))



def test_preload_adds_a_constant_extending_torque(container: ModuleContainer) -> None:
    module = AntagonisticSpring({"spring_constants": [0.1, 0.2], "spring_preloads": [0.01, 0.0]})
    module.finalize(container)
    module.post_finalize(container)
    module.step(container)

    # force = -(spring_constants * joint_pos + spring_preloads) = -([0.1, 0.2] * [1.0, -2.0] + [0.01, 0.0]).
    assert container.antagonistic_spring_force_buf[0, 0].item() == pytest.approx(-0.11)
    assert container.antagonistic_spring_force_buf[0, 1].item() == pytest.approx(0.4)


@pytest.mark.parametrize("config", [{"spring_constants": 0.1, "spring_preloads": -0.01}, {"spring_constants": [-0.1, 0.1]}])
def test_negative_spring_or_preload_is_rejected(container: ModuleContainer, config) -> None:
    module = AntagonisticSpring(config)
    module.finalize(container)
    with pytest.raises(RuntimeError, match="non-negative"):
        module.post_finalize(container)


def test_per_joint_dict_with_default(container: ModuleContainer) -> None:
    module = AntagonisticSpring({"spring_constants": {"default": 0.1, "motor_1": 0.2}, "spring_preloads": {"motor_0": 0.15}})
    module.finalize(container)
    module.post_finalize(container)
    assert module.spring_constants.tolist() == pytest.approx([0.1, 0.2])
    assert module.spring_preloads.tolist() == pytest.approx([0.15, 0.0])


def test_per_joint_dict_rejects_unknown_joints(container: ModuleContainer) -> None:
    module = AntagonisticSpring({"spring_constants": {"Thumb_CMC9": 0.1}})
    module.finalize(container)
    with pytest.raises(RuntimeError, match="Thumb_CMC9"):
        module.post_finalize(container)

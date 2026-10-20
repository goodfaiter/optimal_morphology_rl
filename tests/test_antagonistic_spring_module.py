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


def _make_module(spring_constants=0.1, spring_constant_range=None) -> AntagonisticSpring:
    cfg = {"spring_constants": spring_constants}
    if spring_constant_range is not None:
        cfg["spring_constant_range"] = spring_constant_range
    return AntagonisticSpring(cfg)


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


# ---------------------------------------------------------------------------
# spring_constant_range domain randomization
# ---------------------------------------------------------------------------
def test_range_pins_default_constants_during_training() -> None:
    num_envs = 4
    cont = _make_envs(num_envs, joint_pos=torch.tensor([[1.0, -2.0]] * num_envs))
    module = _make_module(0.1, spring_constant_range=[0.1, 0.1])
    module.finalize(cont)
    module.post_finalize(cont)

    assert module.spring_constants_buf.shape == (num_envs, 2)
    assert torch.allclose(module.spring_constants_buf, torch.full((num_envs, 2), 0.1))

    cont.reset_buf = torch.ones(num_envs, dtype=torch.bool)
    module.reset(cont)
    module.step(cont)

    assert torch.allclose(module.spring_constants_buf, torch.full((num_envs, 2), 0.1))
    assert torch.allclose(cont.antagonistic_spring_force_buf, -0.1 * cont.robot.get_joint_pos_buf)


def test_range_samples_within_bounds_and_step_uses_per_env_constants() -> None:
    num_envs = 8
    cont = _make_envs(num_envs, joint_pos=torch.tensor([[1.0, -2.0]] * num_envs))
    module = _make_module(0.1, spring_constant_range=[0.05, 0.2])
    module.finalize(cont)
    torch.manual_seed(0)
    module.post_finalize(cont)

    constants = module.spring_constants_buf
    assert constants.shape == (num_envs, 2)
    assert torch.all(constants >= 0.05)
    assert torch.all(constants <= 0.2)
    assert (constants.max() - constants.min()).item() > 0.0

    module.step(cont)
    assert torch.allclose(cont.antagonistic_spring_force_buf, -constants * cont.robot.get_joint_pos_buf)


def test_reset_resamples_only_resetting_envs() -> None:
    num_envs = 4
    cont = _make_envs(num_envs)
    module = _make_module(0.1, spring_constant_range=[0.05, 0.2])
    module.finalize(cont)
    torch.manual_seed(0)
    module.post_finalize(cont)

    before = module.spring_constants_buf.clone()
    cont.reset_buf = torch.tensor([True, False, False, False])
    torch.manual_seed(1)
    module.reset(cont)

    after = module.spring_constants_buf
    assert torch.all(after[0] >= 0.05)
    assert torch.all(after[0] <= 0.2)
    assert not torch.allclose(after[0], before[0])
    assert torch.allclose(after[1:], before[1:])


def test_reset_resamples_all_envs_when_reset_buf_all_true() -> None:
    num_envs = 4
    cont = _make_envs(num_envs)
    module = _make_module(0.1, spring_constant_range=[0.05, 0.2])
    module.finalize(cont)
    torch.manual_seed(0)
    module.post_finalize(cont)

    cont.reset_buf = torch.ones(num_envs, dtype=torch.bool)
    module.reset(cont)

    constants = module.spring_constants_buf
    assert torch.all(constants >= 0.05)
    assert torch.all(constants <= 0.2)


def test_reset_is_noop_when_nothing_resets() -> None:
    cont = _make_envs(4)
    module = _make_module(0.1, spring_constant_range=[0.05, 0.2])
    module.finalize(cont)
    torch.manual_seed(0)
    module.post_finalize(cont)

    before = module.spring_constants_buf.clone()
    cont.reset_buf = torch.zeros(4, dtype=torch.bool)
    module.reset(cont)

    assert torch.allclose(module.spring_constants_buf, before)


def test_reset_keeps_defaults_without_range() -> None:
    num_envs = 4
    cont = _make_envs(num_envs, joint_pos=torch.tensor([[1.0, -2.0]] * num_envs))
    module = _make_module(0.1)
    module.finalize(cont)
    module.post_finalize(cont)

    cont.reset_buf = torch.ones(num_envs, dtype=torch.bool)
    module.reset(cont)
    module.step(cont)

    assert torch.allclose(module.spring_constants_buf, 0.1 * torch.ones(num_envs, 2))
    assert torch.allclose(cont.antagonistic_spring_force_buf, -0.1 * cont.robot.get_joint_pos_buf)


def test_post_finalize_supports_range_only_config() -> None:
    num_envs = 4
    cont = _make_envs(num_envs, joint_pos=torch.tensor([[1.0, -2.0]] * num_envs))
    module = AntagonisticSpring({"spring_constant_range": [0.05, 0.2]})
    module.finalize(cont)
    torch.manual_seed(0)
    module.post_finalize(cont)

    assert module.spring_constants is None

    constants = module.spring_constants_buf
    assert constants.shape == (num_envs, 2)
    assert torch.all(constants >= 0.05)
    assert torch.all(constants <= 0.2)

    module.step(cont)
    assert torch.allclose(cont.antagonistic_spring_force_buf, -constants * cont.robot.get_joint_pos_buf)


def test_post_finalize_rejects_min_greater_than_max(container: ModuleContainer) -> None:
    module = _make_module(0.1, spring_constant_range=[0.2, 0.05])
    module.finalize(container)
    with pytest.raises(RuntimeError, match="spring_constant_range"):
        module.post_finalize(container)


def test_post_finalize_rejects_malformed_range(container: ModuleContainer) -> None:
    module = _make_module(0.1, spring_constant_range=0.5)
    module.finalize(container)
    with pytest.raises(RuntimeError, match="spring_constant_range"):
        module.post_finalize(container)

"""Unit tests for the rigid_tendons module."""

import pytest
import torch

from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.rigid_tendons_module import RigidTendons


class _FakeEnv:
    pass


class _FakeRobot:
    def __init__(self, num_tendons: int = 3):
        self.use_tendon = True
        self.num_tendons = num_tendons
        self.get_tendon_lengths_buf = torch.zeros((1, num_tendons), dtype=torch.float32)
        self.get_tendon_vel_buf = torch.zeros((1, num_tendons), dtype=torch.float32)


@pytest.fixture
def container() -> ModuleContainer:
    cont = ModuleContainer()
    cont.total_num_envs = 1
    cont.device = torch.device("cpu")
    cont.env = _FakeEnv()
    cont.robot = _FakeRobot()
    cont.reset_buf = torch.zeros(1, dtype=torch.bool)
    cont.robot.get_tendon_lengths_buf = torch.tensor([[0.0665, 0.06, 0.0865]])
    cont.robot.get_tendon_vel_buf = torch.tensor([[0.0, 0.1, 0.2]])
    return cont


def _make_module(**overrides) -> RigidTendons:
    config = {
        "rigid_tendon_indices": [-1],
        "rest_length": 0.0665,
        "stretch_kp": 10000.0,
        "stretch_kd": 0.0,
    }
    config.update(overrides)
    return RigidTendons(config)


def test_post_finalize_allocates_indices_and_buffer(container: ModuleContainer) -> None:
    module = _make_module()
    module.finalize(container)
    module.post_finalize(container)

    assert container.rigid_tendon_force_buf.shape == (1, 3)
    assert torch.allclose(container.rigid_tendon_indices, torch.tensor([-1]))


def test_step_writes_restoring_force_for_rigid_tendons_only(container: ModuleContainer) -> None:
    module = _make_module()
    module.finalize(container)
    module.post_finalize(container)
    module.step(container)

    # DIP tendon (last column): length 0.0865, rest 0.0665 -> stretch 0.02, force 0.02 * 10000 = 200.
    assert container.rigid_tendon_force_buf[:, -1].item() == pytest.approx(200.0)
    # Only rigid tendon columns are written; the rest stay zero.
    assert container.rigid_tendon_force_buf[:, 0].item() == 0.0
    assert container.rigid_tendon_force_buf[:, 1].item() == 0.0


def test_step_applies_velocity_damping(container: ModuleContainer) -> None:
    module = _make_module(stretch_kd=1.0)
    module.finalize(container)
    module.post_finalize(container)
    module.step(container)

    # force = stretch * kp - vel * kd = 200 - 0.2 * 1.0 = 199.8.
    assert container.rigid_tendon_force_buf[:, -1].item() == pytest.approx(199.8)


def test_step_clamps_unstretched_tendons_to_zero(container: ModuleContainer) -> None:
    module = _make_module()
    module.finalize(container)
    module.post_finalize(container)

    # First tendon is shorter than its rest length: stretch clamps to 0, force clamps to >= 0.
    container.robot.get_tendon_lengths_buf = torch.tensor([[0.05, 0.06, 0.0665]])
    container.robot.get_tendon_vel_buf = torch.tensor([[10.0, 0.0, 0.0]])
    module.step(container)

    assert container.rigid_tendon_force_buf[:, -1].item() == 0.0


def test_step_supports_multiple_rigid_tendons(container: ModuleContainer) -> None:
    module = _make_module(rigid_tendon_indices=[-1, -2])
    module.finalize(container)
    module.post_finalize(container)
    module.step(container)

    # Last tendon: 0.0865/0.0? stretch 0.02 -> force 200; middle tendon: 0.06 - 0.0665 < 0 -> 0.
    assert container.rigid_tendon_force_buf[:, -1].item() == pytest.approx(200.0)
    assert container.rigid_tendon_force_buf[:, -2].item() == 0.0
    assert container.rigid_tendon_force_buf[:, 0].item() == 0.0


def test_post_finalize_rejects_per_index_length_mismatch(container: ModuleContainer) -> None:
    module = _make_module(stretch_kp=[10000.0, 20000.0])  # one index, two values
    module.finalize(container)
    with pytest.raises(RuntimeError, match="stretch_kp"):
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


def test_finalize_requires_tendon_robot(container: ModuleContainer) -> None:
    container.robot.use_tendon = False
    module = _make_module()
    with pytest.raises(RuntimeError, match="use_tendon"):
        module.finalize(container)


def test_finalize_validates_rigid_tendon_indices(container: ModuleContainer) -> None:
    module = _make_module(rigid_tendon_indices=[])
    with pytest.raises(RuntimeError, match="rigid_tendon_indices"):
        module.finalize(container)

    module = _make_module(rigid_tendon_indices=[3])
    with pytest.raises(RuntimeError, match="rigid_tendon_indices"):
        module.finalize(container)

    module = _make_module(rigid_tendon_indices=[-4])
    with pytest.raises(RuntimeError, match="rigid_tendon_indices"):
        module.finalize(container)

    module = _make_module(rigid_tendon_indices=[1.0])  # type: ignore[list-item]
    with pytest.raises(RuntimeError, match="rigid_tendon_indices"):
        module.finalize(container)

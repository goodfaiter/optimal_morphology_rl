"""Unit tests for the randomize_antagonistic_springs module."""

from types import SimpleNamespace

import pytest
import torch

from optimal_morphology_rl.modules.antagonistic_spring_module import AntagonisticSpring
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.randomize_antagonistic_springs_module import RandomizeAntagonisticSpringsModule

NUM_ENVS = 64


class _Manager:
    def __init__(self, modules):
        self.modules = modules

    def __contains__(self, name):
        return name in self.modules

    def get(self, name):
        return self.modules[name]


def _make(scale, spring_constants=(0.1, 1.0, 1.0)):
    """MCP / PIP / DIP springs plus a prismatic slider motor (no spring)."""
    robot = SimpleNamespace(
        use_tendon=True,
        num_motors=4,
        art_def=SimpleNamespace(get_motor_def=lambda i: SimpleNamespace(name=f"motor_{i}")),
        motor_to_joint_dof_index=torch.arange(4),
        revolute_dof_mask=torch.tensor([True, True, True, False]),
        get_joint_pos_buf=torch.zeros(NUM_ENVS, 4),
    )
    container = ModuleContainer()
    container.total_num_envs, container.device, container.robot = NUM_ENVS, torch.device("cpu"), robot
    container.reset_buf = torch.ones(NUM_ENVS, dtype=torch.bool)
    spring = AntagonisticSpring({"spring_constants": list(spring_constants)})
    container.env = SimpleNamespace(module_manager=_Manager({"antagonistic_spring": spring}))
    spring.finalize(container)
    spring.post_finalize(container)
    return RandomizeAntagonisticSpringsModule({"scale": scale}), spring, container


def test_reset_draws_each_constant_within_the_scale() -> None:
    module, spring, container = _make(0.1)
    module.reset(container)
    nominal = torch.tensor([0.1, 1.0, 1.0])
    ratio = spring.spring_constants_buf[:, :3] / nominal
    assert torch.all((ratio >= 0.9 - 1e-6) & (ratio <= 1.1 + 1e-6))
    assert torch.unique(ratio).numel() > NUM_ENVS  # independent per env and motor
    assert torch.all(spring.spring_constants_buf[:, 3] == 0.0)  # the slider has no spring
    assert torch.allclose(spring.spring_constants, torch.tensor([0.1, 1.0, 1.0, 0.0]))  # the nominal is kept


def test_partial_reset_only_changes_resetting_envs() -> None:
    module, spring, container = _make(0.1)
    module.reset(container)
    before = spring.spring_constants_buf.clone()
    container.reset_buf[:] = False
    container.reset_buf[[3, 10]] = True
    module.reset(container)
    changed = (spring.spring_constants_buf != before).any(1)
    assert changed.nonzero().flatten().tolist() == [3, 10]


def test_scale_zero_keeps_the_nominal_constants() -> None:
    module, spring, container = _make(0.0)
    module.reset(container)
    assert torch.allclose(spring.spring_constants_buf, torch.tensor([[0.1, 1.0, 1.0, 0.0]] * NUM_ENVS))


def test_missing_spring_module_raises() -> None:
    module, _, container = _make(0.1)
    container.env = SimpleNamespace(module_manager=_Manager({}))
    with pytest.raises(RuntimeError, match="antagonistic_spring"):
        module.reset(container)


@pytest.mark.parametrize("scale", [-0.1, 1.5])
def test_invalid_scale_raises(scale) -> None:
    with pytest.raises(RuntimeError, match="scale"):
        RandomizeAntagonisticSpringsModule({"scale": scale})

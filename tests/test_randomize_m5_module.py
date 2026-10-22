"""Unit tests for the randomize_m5 module (per-env friction scale of 'apply_m5_forces')."""

from types import SimpleNamespace

import pytest
import torch

from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.randomize_m5_module import RandomizeM5Module

NUM_ENVS = 64


class _Manager:
    def __init__(self, modules):
        self.modules = modules

    def __contains__(self, name):
        return name in self.modules

    def get(self, name):
        return self.modules[name]


def _make(config):
    m5 = SimpleNamespace(friction_scale=torch.ones(NUM_ENVS, 1))
    container = ModuleContainer()
    container.env = SimpleNamespace(module_manager=_Manager({"apply_m5_forces": m5}))
    container.reset_buf = torch.ones(NUM_ENVS, dtype=torch.bool)
    return RandomizeM5Module(config), container, m5


def test_reset_draws_one_factor_per_env_within_the_scale() -> None:
    module, container, m5 = _make({"scale": 0.25})
    module.reset(container)
    factor = m5.friction_scale
    assert factor.shape == (NUM_ENVS, 1)
    assert torch.all((factor >= 0.75 - 1e-6) & (factor <= 1.25 + 1e-6))
    assert torch.unique(factor).numel() == NUM_ENVS  # independent per env


def test_partial_reset_only_redraws_the_resetting_envs() -> None:
    module, container, m5 = _make({"scale": 0.25})
    module.reset(container)
    before = m5.friction_scale.clone()
    container.reset_buf[:] = False
    container.reset_buf[[3, 10]] = True
    module.reset(container)
    changed = (m5.friction_scale != before).any(1)
    assert changed.nonzero().flatten().tolist() == [3, 10]


def test_scale_zero_keeps_the_nominal_friction() -> None:
    module, container, m5 = _make({"scale": 0.0})
    module.reset(container)
    assert torch.all(m5.friction_scale == 1.0)


def test_missing_m5_module_raises() -> None:
    module, container, _ = _make({"scale": 0.1})
    container.env = SimpleNamespace(module_manager=_Manager({}))
    with pytest.raises(RuntimeError, match="apply_m5_forces"):
        module.reset(container)


@pytest.mark.parametrize("scale", [-0.1, 1.5])
def test_invalid_scale_raises(scale) -> None:
    with pytest.raises(RuntimeError, match="scale"):
        RandomizeM5Module({"scale": scale})

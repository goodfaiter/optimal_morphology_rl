"""Unit tests for the randomize_spool_velocity module (spool speed limit of 'apply_m5_forces')."""

from types import SimpleNamespace

import pytest
import torch

from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.randomize_spool_velocity_module import RandomizeSpoolVelocityModule

NUM_ENVS, NUM_TENDONS = 64, 2
RANGE = [6.4, 8.0]


class _Manager:
    def __init__(self, modules):
        self.modules = modules

    def __contains__(self, name):
        return name in self.modules

    def get(self, name):
        return self.modules[name]


def _make(config, max_spool_velocity=7.6):
    limit = None if max_spool_velocity is None else torch.full((NUM_ENVS, NUM_TENDONS), max_spool_velocity)
    m5 = SimpleNamespace(max_spool_velocity=limit)
    container = ModuleContainer()
    container.env = SimpleNamespace(module_manager=_Manager({"apply_m5_forces": m5}))
    container.reset_buf = torch.ones(NUM_ENVS, dtype=torch.bool)
    return RandomizeSpoolVelocityModule(config), container, m5


def test_full_reset_draws_within_the_range_per_env_and_tendon() -> None:
    module, container, m5 = _make({"max_spool_velocity_range": RANGE})
    module.reset(container)
    v = m5.max_spool_velocity
    assert torch.all((v >= RANGE[0]) & (v <= RANGE[1]))
    assert torch.unique(v).numel() > NUM_ENVS


def test_only_resetting_envs_are_resampled() -> None:
    module, container, m5 = _make({"max_spool_velocity_range": RANGE})
    container.reset_buf[: NUM_ENVS // 2] = False
    module.reset(container)
    assert torch.all(m5.max_spool_velocity[: NUM_ENVS // 2] == 7.6)
    assert not torch.any(m5.max_spool_velocity[NUM_ENVS // 2 :] == 7.6)


@pytest.mark.parametrize("value", [None, [0.04], [0.0, 0.04], [0.04, 0.03]])
def test_invalid_range_raises(value) -> None:
    with pytest.raises(RuntimeError, match="max_spool_velocity_range"):
        RandomizeSpoolVelocityModule({"max_spool_velocity_range": value})


def test_requires_a_configured_limit() -> None:
    module, container, _ = _make({"max_spool_velocity_range": RANGE}, max_spool_velocity=None)
    with pytest.raises(RuntimeError, match="max_spool_velocity"):
        module.reset(container)


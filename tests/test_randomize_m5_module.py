"""Unit tests for the randomize_m5 module (M5 servo parameters of 'apply_m5_forces')."""

from types import SimpleNamespace

import pytest
import torch

from optimal_morphology_rl.modules.apply_control.apply_m5_forces_module import M5_PARAMS
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.randomize_m5_module import DEFAULT_PARAMS, RandomizeM5Module

NUM_ENVS, NUM_TENDONS = 64, 2


class _Manager:
    def __init__(self, modules):
        self.modules = modules

    def __contains__(self, name):
        return name in self.modules

    def get(self, name):
        return self.modules[name]


def _m5(friction_model=None):
    m5 = SimpleNamespace(friction_model=friction_model, nominal={})
    for i, (key, attr) in enumerate(M5_PARAMS.items()):
        m5.nominal[key] = torch.tensor([0.01 * (i + 1), 0.02 * (i + 1)])
        setattr(m5, attr, m5.nominal[key].expand(NUM_ENVS, NUM_TENDONS).clone())
    return m5


def _make(config, friction_model=None):
    m5 = _m5(friction_model)
    container = ModuleContainer()
    container.env = SimpleNamespace(module_manager=_Manager({"apply_m5_forces": m5}))
    container.reset_buf = torch.ones(NUM_ENVS, dtype=torch.bool)
    return RandomizeM5Module(config), container, m5


def _ratio(m5, key):
    return getattr(m5, M5_PARAMS[key]) / m5.nominal[key]


def test_full_reset_draws_every_default_parameter_within_the_scale() -> None:
    module, container, m5 = _make({"scale": 0.1})
    module.reset(container)
    for key in DEFAULT_PARAMS:
        ratio = _ratio(m5, key)
        assert torch.all((ratio >= 0.9 - 1e-6) & (ratio <= 1.1 + 1e-6)), key
        assert torch.unique(ratio).numel() > NUM_ENVS  # independent per env and tendon
    assert torch.all(_ratio(m5, "torque_constant") == 1.0)  # not in the defaults


def test_parameters_are_drawn_independently() -> None:
    module, container, m5 = _make({"scale": 0.1})
    module.reset(container)
    assert not torch.equal(_ratio(m5, "K_v"), _ratio(m5, "K_c"))


def test_partial_reset_only_changes_resetting_envs() -> None:
    module, container, m5 = _make({"scale": 0.1})
    module.reset(container)
    before = m5.k_c.clone()
    container.reset_buf[:] = False
    container.reset_buf[[3, 10]] = True
    module.reset(container)
    changed = (m5.k_c != before).any(1)
    assert changed.nonzero().flatten().tolist() == [3, 10]


def test_scale_zero_keeps_the_nominal_values() -> None:
    module, container, m5 = _make({"scale": 0.0})
    module.reset(container)
    for key in DEFAULT_PARAMS:
        assert torch.allclose(_ratio(m5, key), torch.ones(NUM_ENVS, NUM_TENDONS))


def test_explicit_params_including_a_motor_constant() -> None:
    module, container, m5 = _make({"scale": 0.2, "params": ["torque_constant"]})
    module.reset(container)
    ratio = _ratio(m5, "torque_constant")
    assert torch.all((ratio >= 0.8 - 1e-6) & (ratio <= 1.2 + 1e-6)) and not torch.all(ratio == 1.0)
    assert torch.all(_ratio(m5, "K_c") == 1.0)


def _with_model(m5):
    m5.friction_model_scale = torch.ones(NUM_ENVS, NUM_TENDONS)
    return m5


def test_friction_model_envelope_is_scaled_within_the_scale() -> None:
    module, container, m5 = _make({"scale": 0.1}, friction_model=object())
    _with_model(m5)
    module.reset(container)
    factor = m5.friction_model_scale
    assert torch.all((factor >= 0.9 - 1e-6) & (factor <= 1.1 + 1e-6))
    assert torch.unique(factor).numel() > NUM_ENVS  # independent per env and tendon
    for key in DEFAULT_PARAMS:  # the constants are unused with a model and stay nominal
        assert torch.all(_ratio(m5, key) == 1.0), key


def test_friction_model_scale_zero_keeps_the_envelope() -> None:
    module, container, m5 = _make({"scale": 0.0}, friction_model=object())
    _with_model(m5)
    module.reset(container)
    assert torch.all(m5.friction_model_scale == 1.0)


def test_friction_model_partial_reset_and_motor_constant() -> None:
    module, container, m5 = _make({"scale": 0.2, "params": ["torque_constant", "K_c"]}, friction_model=object())
    _with_model(m5)
    module.reset(container)
    assert not torch.all(_ratio(m5, "torque_constant") == 1.0)
    assert torch.all(_ratio(m5, "K_c") == 1.0)  # a friction constant: unused with a model
    before = m5.friction_model_scale.clone()
    container.reset_buf[:] = False
    container.reset_buf[[2, 7]] = True
    module.reset(container)
    changed = (m5.friction_model_scale != before).any(1)
    assert changed.nonzero().flatten().tolist() == [2, 7]


def test_missing_m5_module_raises() -> None:
    module, container, _ = _make({"scale": 0.1})
    container.env = SimpleNamespace(module_manager=_Manager({}))
    with pytest.raises(RuntimeError, match="apply_m5_forces"):
        module.reset(container)


@pytest.mark.parametrize("config, match", [({"scale": 1.0}, "scale"), ({"scale": -0.1}, "scale"), ({"params": ["K_x"]}, "K_x")])
def test_invalid_config_raises(config, match) -> None:
    with pytest.raises(RuntimeError, match=match):
        RandomizeM5Module(config)

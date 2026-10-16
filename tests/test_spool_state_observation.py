"""Unit tests for the spool_state observation."""

from types import SimpleNamespace

import torch

from optimal_morphology_rl.modules.observations.spool_state_observation import SpoolStateObservation


def _spool_env():
    servo = SimpleNamespace(model_tendon_indices=[0, 1], dof_indices=torch.tensor([3, 4]), pully_radius=torch.tensor([0.005, 0.01]))
    robot = SimpleNamespace(
        get_joint_pos_buf=torch.tensor([[9.0, 9.0, 9.0, 0.005, 0.02], [9.0, 9.0, 9.0, 0.0, 0.01]]),
        joint_velocity=torch.tensor([[9.0, 9.0, 9.0, 0.01, -0.01], [9.0, 9.0, 9.0, 0.0, 0.0]]),
    )
    container = SimpleNamespace(robot=robot, spool=servo, get=lambda key: servo if key == "spool" else None)
    manager = SimpleNamespace(container=container)
    return SimpleNamespace(module_manager=manager)


def test_spool_state_dim_is_angle_and_velocity_per_model_tendon() -> None:
    assert SpoolStateObservation().get_obs_dim(_spool_env()) == 4


def test_spool_state_is_slider_state_over_pulley_radius() -> None:
    out = torch.zeros((2, 4))
    SpoolStateObservation().compute_observation(_spool_env(), out)
    expected = torch.tensor([[1.0, 2.0, 2.0, -1.0], [0.0, 1.0, 0.0, 0.0]])
    assert torch.allclose(out, expected)


def _offset_env(reset_buf):
    env = _spool_env()
    container = env.module_manager.container
    spool = container.spool
    values = {"spool": spool, "reset_buf": reset_buf}
    env.module_manager.container = SimpleNamespace(robot=container.robot, spool=spool, get=values.get)
    return env


def test_offset_is_drawn_on_reset_and_added_to_the_angle_only() -> None:
    env = _offset_env(torch.tensor([True, True]))
    module = SpoolStateObservation({"noise": {"offset_range": [-0.2, 0.3]}})
    torch.manual_seed(0)
    module.reset(env)
    assert module.offset.shape == (2, 2)
    assert torch.all((module.offset >= -0.2) & (module.offset <= 0.3)) and torch.unique(module.offset).numel() == 4

    out = torch.zeros((2, 4))
    module.compute_observation(env, out)
    expected = torch.tensor([[1.0, 2.0, 2.0, -1.0], [0.0, 1.0, 0.0, 0.0]])
    assert torch.allclose(out[:, :2], expected[:, :2] + module.offset)
    assert torch.allclose(out[:, 2:], expected[:, 2:])  # velocity unaffected


def test_offset_is_resampled_only_for_resetting_envs() -> None:
    env = _offset_env(torch.tensor([True, True]))
    module = SpoolStateObservation({"noise": {"offset_range": [-0.2, 0.2]}})
    module.reset(env)
    before = module.offset.clone()
    env.module_manager.container.get = {"spool": env.module_manager.container.spool, "reset_buf": torch.tensor([False, True])}.get
    module.reset(env)
    assert torch.equal(module.offset[0], before[0]) and not torch.equal(module.offset[1], before[1])


def test_no_offset_by_default() -> None:
    env = _offset_env(torch.tensor([True, True]))
    module = SpoolStateObservation()
    module.reset(env)
    assert module.offset is None
    out = torch.zeros((2, 4))
    module.compute_observation(env, out)
    assert torch.allclose(out, torch.tensor([[1.0, 2.0, 2.0, -1.0], [0.0, 1.0, 0.0, 0.0]]))


def test_invalid_offset_range_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="offset_range"):
        SpoolStateObservation({"noise": {"offset_range": [0.2, -0.2]}})


def test_observation_manager_forwards_resets_to_its_sub_modules() -> None:
    from optimal_morphology_rl.modules.module_container import ModuleContainer
    from optimal_morphology_rl.modules.observations.observation_manager_module import ObservationManagerModule

    manager = ObservationManagerModule({"modules": {"spool_state": {"noise": {"offset_range": [-0.1, 0.1]}}}})
    assert manager._obs_modules[0].offset_low == -0.1  # the config under modules reaches the sub-module
    calls = []
    manager._obs_modules[0].reset = lambda env: calls.append(env)
    container = ModuleContainer()
    container.env = "env"
    container.reset_buf = torch.tensor([True])
    manager.reset(container)
    assert calls == ["env"]


def test_observation_modules_must_be_a_mapping() -> None:
    import pytest

    from optimal_morphology_rl.modules.observations.observation_manager_module import ObservationManagerModule

    with pytest.raises(ValueError, match="must map each observation sub-module"):
        ObservationManagerModule({"modules": ["spool_state"]})

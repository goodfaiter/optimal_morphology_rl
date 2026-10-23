"""Unit tests for the spool_state observation."""

from types import SimpleNamespace

import torch

from optimal_morphology_rl.modules.observations.spool_state_observation import SpoolStateObservation


def _spool_env():
    servo = SimpleNamespace(model_tendon_indices=[0, 1], dof_indices=torch.tensor([3, 4]), pully_radius=torch.tensor([0.005, 0.01]))
    robot = SimpleNamespace(
        get_joint_pos_buf=torch.tensor([[9.0, 9.0, 9.0, 0.005, 0.02], [9.0, 9.0, 9.0, 0.0, 0.01]]),
        get_joint_vel_buf=torch.tensor([[9.0, 9.0, 9.0, 0.01, -0.01], [9.0, 9.0, 9.0, 0.0, 0.0]]),
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
    spool.dt = 0.005
    values = {"spool": spool, "reset_buf": reset_buf}
    env.module_manager.container = SimpleNamespace(robot=container.robot, spool=spool, get=values.get)
    return env


def _set_reset_buf(env, reset_buf):
    values = {"spool": env.module_manager.container.spool, "reset_buf": reset_buf}
    env.module_manager.container.get = values.get


RAW = torch.tensor([[1.0, 2.0, 2.0, -1.0], [0.0, 1.0, 0.0, 0.0]])  # [theta (2), omega (2)] of _spool_env


def test_angle_offset_is_drawn_on_reset_and_added_to_the_angle() -> None:
    env = _offset_env(torch.tensor([True, True]))
    module = SpoolStateObservation({"noise": {"angle_offset_range": [-0.2, 0.3]}})
    torch.manual_seed(0)
    module.reset(env)
    assert module.angle_offset.shape == (2, 2)
    assert torch.all((module.angle_offset >= -0.2) & (module.angle_offset <= 0.3)) and torch.unique(module.angle_offset).numel() == 4

    out = torch.zeros((2, 4))
    module.compute_observation(env, out)
    assert torch.allclose(out[:, :2], RAW[:, :2] + module.angle_offset)
    assert torch.allclose(out[:, 2:], RAW[:, 2:])  # no velocity noise configured


def test_noise_is_resampled_only_for_resetting_envs() -> None:
    env = _offset_env(torch.tensor([True, True]))
    noise = {"angle_offset_range": [-0.2, 0.2], "velocity_offset_range": [-1.0, 1.0], "velocity_lag_range": [0.0, 0.1]}
    module = SpoolStateObservation({"noise": noise})
    module.reset(env)
    before = [module.angle_offset.clone(), module.velocity_offset.clone(), module.velocity_alpha.clone()]
    _set_reset_buf(env, torch.tensor([False, True]))
    module.reset(env)
    for old, new in zip(before, [module.angle_offset, module.velocity_offset, module.velocity_alpha]):
        assert torch.equal(new[0], old[0]) and not torch.equal(new[1], old[1])


def test_velocity_low_pass_and_offset() -> None:
    env = _offset_env(torch.tensor([True, True]))
    module = SpoolStateObservation({"noise": {"velocity_lag_range": [0.015, 0.015], "velocity_offset_range": [0.5, 0.5]}})
    module.reset(env)
    alpha = 0.005 / (0.015 + 0.005)
    assert torch.allclose(module.velocity_alpha, torch.full((2, 2), alpha))

    robot = env.module_manager.container.robot
    out = torch.zeros((2, 4))
    module.compute_observation(env, out)  # first step after the reset: the filter starts at the measurement
    assert torch.allclose(out[:, 2:], RAW[:, 2:] + 0.5)
    assert torch.allclose(out[:, :2], RAW[:, :2])  # angle unaffected

    robot.get_joint_vel_buf = robot.get_joint_vel_buf * 0.0  # the spools stop
    for k in range(1, 4):
        module.compute_observation(env, out)
        assert torch.allclose(out[:, 2:], RAW[:, 2:] * (1.0 - alpha) ** k + 0.5)  # decays towards 0, lagging


def test_zero_lag_follows_the_measurement() -> None:
    env = _offset_env(torch.tensor([True, True]))
    module = SpoolStateObservation({"noise": {"velocity_lag_range": [0.0, 0.0], "angle_offset_range": [0.1, 0.1]}})
    module.reset(env)
    out = torch.zeros((2, 4))
    module.compute_observation(env, out)
    env.module_manager.container.robot.get_joint_vel_buf = env.module_manager.container.robot.get_joint_vel_buf * 3.0
    module.compute_observation(env, out)
    assert torch.allclose(out[:, 2:], RAW[:, 2:] * 3.0)


def test_no_noise_by_default() -> None:
    env = _offset_env(torch.tensor([True, True]))
    module = SpoolStateObservation()
    module.reset(env)
    assert module.angle_offset is None
    out = torch.zeros((2, 4))
    module.compute_observation(env, out)
    assert torch.allclose(out, RAW)


def test_invalid_noise_config_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="angle_offset_range"):
        SpoolStateObservation({"noise": {"angle_offset_range": [0.2, -0.2]}})
    with pytest.raises(ValueError, match="velocity_lag_range"):
        SpoolStateObservation({"noise": {"velocity_lag_range": [-0.01, 0.01]}})
    with pytest.raises(ValueError, match="unknown keys"):
        SpoolStateObservation({"noise": {"offset_range": [-0.1, 0.1]}})


def test_observation_manager_forwards_resets_to_its_sub_modules() -> None:
    from optimal_morphology_rl.modules.module_container import ModuleContainer
    from optimal_morphology_rl.modules.observations.observation_manager_module import ObservationManagerModule

    manager = ObservationManagerModule({"modules": {"spool_state": {"noise": {"angle_offset_range": [-0.1, 0.1]}}}})
    assert manager._obs_modules[0].ranges["angle_offset_range"] == (-0.1, 0.1)  # the config reaches the sub-module
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

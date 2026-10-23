"""Unit tests for the privileged (critic-only) observations: spool_torques and the observation manager's group."""

from types import SimpleNamespace

import pytest
import torch

from optimal_morphology_rl.envs.modular_environment import ModularEnvironment
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.observations.observation_manager_module import ObservationManagerModule
from optimal_morphology_rl.modules.observations.spool_torques_observation import SpoolTorquesObservation

SIGNALS = {
    "tau_f_max": torch.tensor([[0.01, 0.02], [0.03, 0.04]]),
    "tau_f": torch.tensor([[-0.01, 0.0], [0.0, 0.01]]),
    "tau_m": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
    "tau_e": torch.tensor([[-0.05, -0.06], [-0.07, -0.08]]),
}


def _env(last_signals=SIGNALS):
    container = ModuleContainer()
    container.device, container.total_num_envs = torch.device("cpu"), 2
    container.spool = SimpleNamespace(model_tendon_indices=[0, 1], dof_indices=torch.tensor([3, 4]), pully_radius=torch.tensor([0.005, 0.005]))
    container.robot = SimpleNamespace(get_joint_pos_buf=torch.zeros(2, 5), get_joint_vel_buf=torch.zeros(2, 5))
    m5 = SimpleNamespace(last_signals=last_signals) if last_signals is not None else SimpleNamespace()
    env = SimpleNamespace(device=torch.device("cpu"), total_num_envs=2,
                          module_manager=SimpleNamespace(container=container, get=lambda name: m5))
    container.env = env
    return env, container


def test_spool_torques_default_is_the_friction_budget() -> None:
    env, _ = _env()
    module = SpoolTorquesObservation()
    assert module.get_obs_dim(env) == 2
    out = torch.zeros(2, 2)
    module.compute_observation(env, out)
    assert torch.equal(out, SIGNALS["tau_f_max"])


def test_spool_torques_signals_in_order() -> None:
    env, _ = _env()
    module = SpoolTorquesObservation({"signals": ["tau_e", "tau_f_max"]})
    out = torch.zeros(2, module.get_obs_dim(env))
    module.compute_observation(env, out)
    assert torch.equal(out, torch.cat([SIGNALS["tau_e"], SIGNALS["tau_f_max"]], dim=-1))


def test_spool_torques_are_zero_before_the_first_step() -> None:
    env, _ = _env(last_signals=None)
    out = torch.ones(2, 2)
    SpoolTorquesObservation().compute_observation(env, out)
    assert torch.all(out == 0.0)


@pytest.mark.parametrize("signals", [[], ["tau_x"]])
def test_spool_torques_invalid_signals_raise(signals) -> None:
    with pytest.raises(ValueError, match="signals"):
        SpoolTorquesObservation({"signals": signals})


def test_observation_manager_builds_the_privileged_observation() -> None:
    env, container = _env()
    manager = ObservationManagerModule({"modules": {"spool_state": {}}, "privileged_modules": {"spool_torques": {}}})
    manager.finalize(container)
    assert env.observation_space.shape == (4,)  # spool_state: theta + omega per spool
    assert env.privileged_observation_space.shape == (2,)  # tau_f_max per spool
    manager.post_finalize(container)
    env.obs_buf = container.obs_buf
    manager.step(container)
    assert torch.equal(container.priv_obs_buf, SIGNALS["tau_f_max"])


def test_observation_manager_without_privileged_modules_has_no_privileged_space() -> None:
    env, container = _env()
    manager = ObservationManagerModule({"modules": {"spool_state": {}}})
    manager.finalize(container)
    manager.post_finalize(container)
    assert not hasattr(env, "privileged_observation_space")
    assert container.get("priv_obs_buf") is None


def test_env_returns_the_privileged_observations_as_a_dict() -> None:
    env = ModularEnvironment.__new__(ModularEnvironment)
    container = ModuleContainer()
    container.obs_buf = torch.ones(2, 4)
    env.module_manager = SimpleNamespace(container=container)
    assert torch.equal(env._observations(), container.obs_buf)  # no privileged group: the plain tensor
    container.priv_obs_buf = torch.full((2, 2), 3.0)
    obs = env._observations()
    assert set(obs) == {"obs", "priv_obs"}
    assert torch.equal(obs["obs"], container.obs_buf) and torch.equal(obs["priv_obs"], container.priv_obs_buf)

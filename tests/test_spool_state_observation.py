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
    manager = SimpleNamespace(get=lambda name: servo, container=SimpleNamespace(robot=robot))
    return SimpleNamespace(module_manager=manager)


def test_spool_state_dim_is_angle_and_velocity_per_model_tendon() -> None:
    assert SpoolStateObservation().get_obs_dim(_spool_env()) == 4


def test_spool_state_is_slider_state_over_pulley_radius() -> None:
    out = torch.zeros((2, 4))
    SpoolStateObservation().compute_observation(_spool_env(), out)
    expected = torch.tensor([[1.0, 2.0, 2.0, -1.0], [0.0, 1.0, 0.0, 0.0]])
    assert torch.allclose(out, expected)

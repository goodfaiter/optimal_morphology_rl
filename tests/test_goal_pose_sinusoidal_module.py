"""Unit tests for the goal_pose_sinusoidal module."""

import pytest
import torch

from optimal_morphology_rl.modules.goal_pose_sinusoidal_module import GoalPoseSinusoidalModule
from optimal_morphology_rl.modules.module_container import ModuleContainer


def _config(**overrides) -> dict:
    config = {
        "goal_position": {"x": [0.0, 0.1], "y": [-0.15, -0.15], "z": [0.2, 0.3]},
        "goal_orientation": [0.0, 0.0, 0.0, 1.0],
        "sinusoidal": {"min_frequency": 0.2, "max_frequency": 0.8},
    }
    config.update(overrides)
    return config


def _container(total_num_envs: int = 4) -> ModuleContainer:
    container = ModuleContainer()
    container.total_num_envs = total_num_envs
    container.device = torch.device("cpu")
    container.timestep = 0.005
    container.frame_skip = 2
    container.reset_buf = torch.zeros(total_num_envs, dtype=torch.bool)
    return container


def test_post_finalize_allocates_goal_and_sinusoid_buffers() -> None:
    container = _container(4)
    module = GoalPoseSinusoidalModule(_config())

    module.post_finalize(container)

    assert container.pos_goal_in_world.shape == (4, 3)
    assert container.quat_goal_to_world.shape == (4, 4)
    assert module._time_buf.shape == (4,)
    assert module._freq.shape == (4, 3)
    assert module._phase.shape == (4, 3)
    assert module._amp.shape == (4, 3)
    assert module._dt == pytest.approx(0.01)


def test_reset_draws_pattern_and_sets_t0_goal() -> None:
    container = _container(8)
    container.reset_buf.fill_(True)
    module = GoalPoseSinusoidalModule(_config())

    module.reset(container)

    half = module._half_range
    assert (module._freq >= 0.2 - 1e-6).all() and (module._freq <= 0.8 + 1e-6).all()
    assert (module._amp >= 0.3 * half - 1e-7).all() and (module._amp <= 1.0 * half + 1e-7).all()
    assert (module._time_buf == 0.0).all()
    # Surface formula matches the buffers at t=0.
    expected = module._center + module._amp * torch.sin(module._phase)
    assert torch.allclose(container.pos_goal_in_world, expected, atol=1e-6)
    assert torch.allclose(container.quat_goal_to_world, torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 8))


def test_step_advances_continuously_within_ranges() -> None:
    container = _container(8)
    container.reset_buf.fill_(True)
    module = GoalPoseSinusoidalModule(_config())

    module.reset(container)
    lo = module._center - module._half_range
    hi = module._center + module._half_range
    bound = 2 * torch.pi * 0.8 * module._half_range * module._dt + 1e-5

    for i in range(500):
        prev = container.pos_goal_in_world.clone()
        module.step(container)
        delta = (container.pos_goal_in_world - prev).abs()
        assert (delta <= bound).all(), f"goal jumped at step {i}"
        assert delta.max().item() > 0.0, "goal did not move"
        assert (container.pos_goal_in_world >= lo - 1e-6).all()
        assert (container.pos_goal_in_world <= hi + 1e-6).all()


def test_full_reset_redraws_pattern_and_zeroes_time() -> None:
    container = _container(8)
    container.reset_buf.fill_(True)
    module = GoalPoseSinusoidalModule(_config())

    module.reset(container)
    module._time_buf.fill_(10.0)
    old_freq = module._freq.clone()
    old_pos = container.pos_goal_in_world.clone()

    module.reset(container)

    assert (module._time_buf == 0.0).all()
    assert not torch.allclose(module._freq, old_freq), "pattern was not redrawn on reset"
    assert not torch.allclose(container.pos_goal_in_world, old_pos)


def test_partial_reset_only_changes_reset_envs() -> None:
    container = _container(4)
    container.reset_buf.fill_(True)
    module = GoalPoseSinusoidalModule(_config())

    module.reset(container)
    first_freq = module._freq.clone()
    first_quat = container.quat_goal_to_world.clone()
    container.reset_buf[:] = False
    container.reset_buf[[1, 3]] = True
    module.reset(container)

    assert torch.allclose(first_freq[0], module._freq[0])
    assert not torch.allclose(first_freq[1], module._freq[1])
    assert torch.allclose(first_freq[2], module._freq[2])
    assert not torch.allclose(first_freq[3], module._freq[3])
    assert (module._time_buf[[1, 3]] == 0.0).all()
    # Orientation redrawn for reset envs, kept for the others.
    assert torch.allclose(first_quat[0], container.quat_goal_to_world[0])
    assert torch.allclose(first_quat[2], container.quat_goal_to_world[2])


def test_orientation_random_samples_unit_quaternions() -> None:
    container = _container(16)
    container.reset_buf.fill_(True)
    module = GoalPoseSinusoidalModule(_config(goal_orientation="random"))

    module.reset(container)

    goal_quat = container.quat_goal_to_world
    assert torch.allclose(torch.linalg.vector_norm(goal_quat, dim=-1), torch.ones(16), atol=1e-5)


def test_degenerate_y_range_gives_zero_y_amplitude() -> None:
    container = _container(4)
    container.reset_buf.fill_(True)
    module = GoalPoseSinusoidalModule(_config())

    module.reset(container)

    assert module._half_range[1].item() == pytest.approx(0.0)
    assert container.pos_goal_in_world[:, 1] == pytest.approx(module._center[1].item(), abs=1e-7)


def test_invalid_frequency_config_raises() -> None:
    with pytest.raises(ValueError, match="min_frequency"):
        GoalPoseSinusoidalModule(_config(sinusoidal={"min_frequency": 0.0, "max_frequency": 0.8}))

    with pytest.raises(ValueError, match="min_frequency"):
        GoalPoseSinusoidalModule(_config(sinusoidal={"min_frequency": 0.9, "max_frequency": 0.8}))


def test_invalid_amplitude_fraction_config_raises() -> None:
    with pytest.raises(ValueError, match="amplitude_fraction"):
        GoalPoseSinusoidalModule(
            _config(sinusoidal={"min_amplitude_fraction": 1.2, "max_amplitude_fraction": 1.5})
        )

    with pytest.raises(ValueError, match="amplitude_fraction"):
        GoalPoseSinusoidalModule(
            _config(sinusoidal={"min_amplitude_fraction": 0.8, "max_amplitude_fraction": 0.2})
        )


def test_missing_timestep_raises_at_allocation() -> None:
    container = ModuleContainer()
    container.total_num_envs = 4
    container.device = torch.device("cpu")
    container.reset_buf = torch.zeros(4, dtype=torch.bool)
    module = GoalPoseSinusoidalModule(_config())

    with pytest.raises(RuntimeError, match="timestep"):
        module.reset(container)

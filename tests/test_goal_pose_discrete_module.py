"""Unit tests for the goal_pose_discrete module."""

import pytest
import torch

from optimal_morphology_rl.modules.goal_pose_discrete_module import GoalPoseDiscreteModule
from optimal_morphology_rl.modules.module_container import ModuleContainer


def _config(**overrides) -> dict:
    config = {
        "goal_position": {"x": [0.0, 0.1], "y": [-0.15, -0.15], "z": [0.2, 0.3]},
        "goal_orientation": [0.0, 0.0, 0.0, 1.0],
        "resample_prob": 0.0,
    }
    config.update(overrides)
    return config


def _container(total_num_envs: int = 1) -> ModuleContainer:
    container = ModuleContainer()
    container.total_num_envs = total_num_envs
    container.device = torch.device("cpu")
    container.reset_buf = torch.zeros(total_num_envs, dtype=torch.bool)
    return container


def test_post_finalize_allocates_goal_buffers() -> None:
    container = _container(4)
    module = GoalPoseDiscreteModule(_config())

    module.post_finalize(container)

    assert container.pos_goal_in_world.shape == (4, 3)
    assert container.quat_goal_to_world.shape == (4, 4)
    # Only allocation; sampling happens in reset.
    assert torch.allclose(container.pos_goal_in_world, torch.zeros((4, 3)))


def test_reset_samples_within_ranges() -> None:
    container = _container(1)
    container.reset_buf.fill_(True)
    module = GoalPoseDiscreteModule(_config())

    module.reset(container)

    goal_pos = container.pos_goal_in_world
    goal_quat = container.quat_goal_to_world
    assert goal_pos.shape == (1, 3)
    assert goal_quat.shape == (1, 4)
    assert 0.0 <= goal_pos[:, 0].item() <= 0.1
    assert goal_pos[:, 1].item() == pytest.approx(-0.15)
    assert 0.2 <= goal_pos[:, 2].item() <= 0.3
    assert torch.allclose(goal_quat, torch.tensor([[0.0, 0.0, 0.0, 1.0]]))


def test_reset_resamples_on_every_reset_call() -> None:
    container = _container(8)
    container.reset_buf.fill_(True)
    module = GoalPoseDiscreteModule(_config())

    module.reset(container)
    first_pos = container.pos_goal_in_world.clone()
    module.reset(container)

    assert not torch.allclose(first_pos, container.pos_goal_in_world)
    assert torch.all(container.pos_goal_in_world[:, 0] <= 0.1)
    assert torch.allclose(container.pos_goal_in_world[:, 1], torch.full((8,), -0.15))
    assert torch.all(container.pos_goal_in_world[:, 2] >= 0.2)
    assert torch.all(container.pos_goal_in_world[:, 2] <= 0.3)


def test_full_env_reset_resamples_all_envs() -> None:
    container = _container(4)
    container.reset_buf.fill_(True)
    module = GoalPoseDiscreteModule({"goal_position": {"x": [0.0, 0.1], "y": [0, 0], "z": [0, 0]}, "resample_prob": 0.0})

    module.reset(container)
    first = container.pos_goal_in_world.clone()
    module.reset(container)
    second = container.pos_goal_in_world

    for i in range(4):
        assert not torch.isclose(first[i, 0], second[i, 0]).item()


def test_partial_reset_only_changes_reset_envs() -> None:
    container = _container(4)
    container.reset_buf.fill_(True)
    module = GoalPoseDiscreteModule(_config())

    module.reset(container)
    first = container.pos_goal_in_world.clone()
    container.reset_buf[:] = False
    container.reset_buf[[1, 3]] = True
    module.reset(container)

    assert torch.allclose(first[0], container.pos_goal_in_world[0])
    assert not torch.allclose(first[1], container.pos_goal_in_world[1])
    assert torch.allclose(first[2], container.pos_goal_in_world[2])
    assert not torch.allclose(first[3], container.pos_goal_in_world[3])


def test_reset_with_missing_reset_buf_samples_all() -> None:
    container = ModuleContainer()
    container.total_num_envs = 3
    container.device = torch.device("cpu")
    container.reset_buf = None
    module = GoalPoseDiscreteModule(_config())

    module.reset(container)

    goal_pos = container.pos_goal_in_world
    assert goal_pos.shape == (3, 3)
    assert torch.all(goal_pos[:, 0] <= 0.1)
    assert torch.all(goal_pos[:, 2] >= 0.2)


def test_reset_with_lazy_allocation_raises_without_env_buffers() -> None:
    container = ModuleContainer()
    container.reset_buf = torch.zeros(2, dtype=torch.bool)
    container.reset_buf.fill_(True)
    module = GoalPoseDiscreteModule(_config())

    # total_num_envs/device missing entirely.
    with pytest.raises(RuntimeError, match="total_num_envs"):
        module.reset(container)


def test_resample_prob_zero_persists_between_steps() -> None:
    container = _container(4)
    container.reset_buf.fill_(True)
    module = GoalPoseDiscreteModule(_config(resample_prob=0.0))

    module.reset(container)
    first = container.pos_goal_in_world.clone()
    module.step(container)

    assert torch.allclose(container.pos_goal_in_world, first)
    assert torch.allclose(container.quat_goal_to_world, torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 4))


def test_resample_prob_one_resamples_every_step() -> None:
    total_num_envs = 8
    container = _container(total_num_envs)
    module = GoalPoseDiscreteModule(
        {
            "goal_position": {"x": [0.0, 0.1], "y": [0, 0], "z": [0, 0]},
            "resample_prob": 1.0,
        }
    )
    container.reset_buf.fill_(True)

    module.reset(container)
    first = container.pos_goal_in_world.clone()
    module.step(container)

    for i in range(total_num_envs):
        assert not torch.isclose(first[i, 0], container.pos_goal_in_world[i, 0]).item()
    # Still inside the configured box.
    assert torch.all(container.pos_goal_in_world[:, 0] <= 0.1)


def test_resample_prob_mask_semantics(monkeypatch) -> None:
    container = _container(4)
    container.reset_buf.fill_(True)
    module = GoalPoseDiscreteModule(
        {
            "goal_position": {"x": [0.0, 0.1], "y": [0, 0], "z": [0, 0]},
            "resample_prob": 0.01,
        }
    )

    module.reset(container)
    first = container.pos_goal_in_world.clone()

    # torch.rand returning zeros selects every env for the step resample.
    monkeypatch.setattr(torch, "rand", lambda n, device=None: torch.zeros(n, device=device))
    module.step(container)
    for i in range(4):
        assert not torch.isclose(first[i, 0], container.pos_goal_in_world[i, 0]).item()
    assert torch.all(container.pos_goal_in_world[:, 0] <= 0.1)

    # torch.rand returning ones selects no env; the buffer stays unchanged.
    second = container.pos_goal_in_world.clone()
    monkeypatch.setattr(torch, "rand", lambda n, device=None: torch.ones(n, device=device))
    module.step(container)
    assert torch.allclose(second, container.pos_goal_in_world)


def test_goal_orientation_random_samples_unit_quaternions() -> None:
    container = _container(16)
    container.reset_buf.fill_(True)
    module = GoalPoseDiscreteModule(_config(goal_orientation="random"))

    module.reset(container)

    goal_quat = container.quat_goal_to_world
    assert torch.allclose(torch.linalg.vector_norm(goal_quat, dim=-1), torch.ones(16), atol=1e-5)


def test_goal_orientation_fixed_samples_configured_quat() -> None:
    container = _container(4)
    container.reset_buf.fill_(True)
    module = GoalPoseDiscreteModule(_config(goal_orientation=[0.7071068, 0.0, 0.0, 0.7071068]))

    module.reset(container)

    goal_quat = container.quat_goal_to_world
    expected = torch.full((4, 4), 0.0)
    expected[:, 0] = 0.7071068
    expected[:, 3] = 0.7071068
    assert torch.allclose(goal_quat, expected)


def test_degenerate_ranges_sample_fixed_position() -> None:
    container = _container(4)
    container.reset_buf.fill_(True)
    module = GoalPoseDiscreteModule(
        {
            "goal_position": {"x": [0.3, 0.3], "y": [0.0, 0.0], "z": [0.1, 0.1]},
            "resample_prob": 0.0,
        }
    )

    module.reset(container)

    expected = torch.tensor([[0.3, 0.0, 0.1]])
    assert torch.allclose(container.pos_goal_in_world, expected)


def test_defaults_sample_zero_center() -> None:
    container = _container(2)
    container.reset_buf.fill_(True)
    module = GoalPoseDiscreteModule({})

    module.reset(container)

    assert torch.allclose(container.pos_goal_in_world, torch.zeros((2, 3)))
    quat = container.quat_goal_to_world
    assert torch.allclose(torch.linalg.vector_norm(quat, dim=-1), torch.ones(2), atol=1e-5)

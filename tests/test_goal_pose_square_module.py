"""Unit tests for the goal_pose_square module."""

import pytest
import torch

from optimal_morphology_rl.modules.goal_pose_square_module import GoalPoseSquareModule
from optimal_morphology_rl.modules.module_container import ModuleContainer


def _config(**overrides) -> dict:
    config = {
        "goal_position": {"x": [0.0, 0.1], "y": [-0.15, -0.15], "z": [0.2, 0.3]},
        "goal_orientation": [0.0, 0.0, 0.0, 1.0],
        "speed": {"min": 0.01, "max": 0.05},
    }
    config.update(overrides)
    return config


def _container(total_num_envs: int = 4, timestep: float = 0.005, frame_skip: int = 2) -> ModuleContainer:
    container = ModuleContainer()
    container.total_num_envs = total_num_envs
    container.device = torch.device("cpu")
    container.timestep = timestep
    container.frame_skip = frame_skip
    container.reset_buf = torch.zeros(total_num_envs, dtype=torch.bool)
    return container


def test_post_finalize_allocates_goal_and_path_buffers() -> None:
    container = _container(4)
    module = GoalPoseSquareModule(_config())

    module.post_finalize(container)

    assert container.pos_goal_in_world.shape == (4, 3)
    assert container.quat_goal_to_world.shape == (4, 4)
    assert module._time_buf.shape == (4,)
    assert module._speed.shape == (4,)
    assert module._start.shape == (4,)
    assert module._dt == pytest.approx(0.01)


def test_default_plane_picks_two_largest_half_ranges() -> None:
    module = GoalPoseSquareModule(_config())
    assert module._plane == (0, 2)


def test_configured_plane_is_respected() -> None:
    module = GoalPoseSquareModule(_config(plane=[1, 2]))
    assert module._plane == (1, 2)
    container = _container(2)
    module.post_finalize(container)
    module._resample(container, None)
    assert torch.allclose(container.pos_goal_in_world[:, 0], torch.full((2,), 0.05))


def test_invalid_plane_config_raises() -> None:
    with pytest.raises(ValueError, match="plane"):
        GoalPoseSquareModule(_config(plane=[0]))

    with pytest.raises(ValueError, match="plane"):
        GoalPoseSquareModule(_config(plane=[1, 1]))

    with pytest.raises(ValueError, match="plane"):
        GoalPoseSquareModule(_config(plane=[0, 3]))


def test_invalid_speed_config_raises() -> None:
    with pytest.raises(ValueError, match="min_speed"):
        GoalPoseSquareModule(_config(speed={"min": 0.0, "max": 0.05}))

    with pytest.raises(ValueError, match="min_speed"):
        GoalPoseSquareModule(_config(speed={"min": 0.1, "max": 0.05}))


def test_reset_draws_speed_and_start_sets_t0_goal() -> None:
    container = _container(8)
    container.reset_buf.fill_(True)
    module = GoalPoseSquareModule(_config())

    module.reset(container)

    assert (module._speed >= 0.01 - 1e-7).all() and (module._speed <= 0.05 + 1e-7).all()
    assert (module._start >= 0.0 - 1e-7).all() and (module._start <= module._perimeter + 1e-7).all()
    assert module._perimeter == pytest.approx(2 * (0.1 + 0.1))
    assert (module._time_buf == 0.0).all()
    pos = container.pos_goal_in_world
    # On the outer edge: at least one plane coordinate at an extreme.
    x_lo, x_hi = module._corners[:, 0].min(), module._corners[:, 0].max()
    z_lo, z_hi = module._corners[:, 1].min(), module._corners[:, 1].max()
    dx = torch.minimum((pos[:, 0] - x_lo).abs(), (pos[:, 0] - x_hi).abs())
    dz = torch.minimum((pos[:, 2] - z_lo).abs(), (pos[:, 2] - z_hi).abs())
    assert ((dx <= 1e-6) | (dz <= 1e-6)).all()
    assert torch.allclose(pos[:, 1], torch.full((8,), module._center_inactive.item()))
    assert torch.allclose(container.quat_goal_to_world, torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 8))


def test_positions_walks_the_perimeter() -> None:
    container = _container(1)
    module = GoalPoseSquareModule(_config())
    module.post_finalize(container)

    w = 0.1
    distances = torch.tensor([0.0, w / 2, w, w + w / 2, w + w, 2 * w + w / 2, 2 * (w + w) - 1e-6])
    pos = module._positions(distances)

    expected = torch.tensor(
        [
            [0.0, -0.15, 0.2],
            [0.05, -0.15, 0.2],
            [0.1, -0.15, 0.2],
            [0.1, -0.15, 0.25],
            [0.1, -0.15, 0.3],
            [0.05, -0.15, 0.3],
            [0.0, -0.15, 0.2],
        ]
    )
    assert torch.allclose(pos, expected, atol=1e-5)


def test_step_advances_continuously_within_ranges() -> None:
    container = _container(8)
    container.reset_buf.fill_(True)
    module = GoalPoseSquareModule(_config())

    module.reset(container)
    lo = module._center_inactive + torch.zeros(3)
    hi = module._center_inactive + torch.zeros(3)
    lo[module._axis0] = module._corners[:, 0].min()
    hi[module._axis0] = module._corners[:, 0].max()
    lo[module._axis1] = module._corners[:, 1].min()
    hi[module._axis1] = module._corners[:, 1].max()
    bound = 0.05 * module._dt + 1e-5

    for i in range(500):
        prev = container.pos_goal_in_world.clone()
        module.step(container)
        delta = (container.pos_goal_in_world - prev).abs().max(-1).values
        assert (delta <= bound).all(), f"goal jumped at step {i}: {delta.max().item()}"
        assert delta.min().item() > 0.0, "goal did not move"
        pos = container.pos_goal_in_world
        for a in range(3):
            assert (pos[:, a] >= lo[a] - 1e-6).all() and (pos[:, a] <= hi[a] + 1e-6).all()


def test_full_reset_redraws_pattern_and_zeroes_time() -> None:
    container = _container(8)
    container.reset_buf.fill_(True)
    module = GoalPoseSquareModule(_config())

    module.reset(container)
    module._time_buf.fill_(10.0)
    old_speed = module._speed.clone()
    old_start = module._start.clone()

    module.reset(container)

    assert (module._time_buf == 0.0).all()
    assert not torch.allclose(module._speed, old_speed)
    assert not torch.allclose(module._start, old_start)


def test_partial_reset_only_changes_reset_envs() -> None:
    container = _container(4)
    container.reset_buf.fill_(True)
    module = GoalPoseSquareModule(_config())

    module.reset(container)
    first_speed = module._speed.clone()
    first_start = module._start.clone()
    container.reset_buf[:] = False
    container.reset_buf[[1, 3]] = True
    module.reset(container)

    assert torch.allclose(first_speed[0], module._speed[0])
    assert not torch.allclose(first_speed[1], module._speed[1])
    assert torch.allclose(first_start[2], module._start[2])
    assert not torch.allclose(first_start[3], module._start[3])
    assert (module._time_buf[[1, 3]] == 0.0).all()


def test_missing_timestep_raises_at_allocation() -> None:
    container = ModuleContainer()
    container.total_num_envs = 4
    container.device = torch.device("cpu")
    container.reset_buf = torch.zeros(4, dtype=torch.bool)
    module = GoalPoseSquareModule(_config())

    with pytest.raises(RuntimeError, match="timestep"):
        module.reset(container)

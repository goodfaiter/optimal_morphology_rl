"""cube_away_termination: distance to the goal and / or a box in the environment frame."""

from types import SimpleNamespace

import pytest
import torch

from optimal_morphology_rl.envs.adapt_cube.terminations import cube_away_termination as cat

BOX = {"x": [-0.055, 0.035], "y": [-0.23, -0.10], "z": [0.115, 0.22]}


def _run(config: dict, pos: list[list[float]], goal: list[float] | None = None, monkeypatch=None) -> torch.Tensor:
    pos_t = torch.tensor(pos)
    n = pos_t.shape[0]
    container = SimpleNamespace(reward_object=SimpleNamespace(name="cube"))
    env = SimpleNamespace(module_manager=SimpleNamespace(container=container), term_buf=torch.zeros(n, dtype=torch.bool), rew_buf=torch.zeros(n))
    monkeypatch.setattr(cat, "find_kinematic_sensor", lambda c, name: SimpleNamespace(pos_in_world=pos_t))
    goal_t = torch.tensor(goal or [0.0, 0.0, 0.0]).expand(n, 3)
    monkeypatch.setattr(cat, "goal_in_env_frame", lambda c: (goal_t, None))
    cat.CubeAwayTermination(config).compute(env)
    return env.term_buf, env.rew_buf


def test_box_terminates_outside_only(monkeypatch) -> None:
    pos = [
        [-0.01, -0.15, 0.145],  # on the fingers
        [-0.075, -0.15, 0.146],  # on the palm
        [-0.01, -0.25, 0.145],  # off the index / thumb side
        [-0.01, -0.15, 0.05],  # fell off
        [-0.01, -0.15, 0.30],  # thrown up
        [0.05, -0.15, 0.145],  # past the fingertips
    ]
    term, rew = _run({"box": BOX, "reward_scale": -10.0}, pos, monkeypatch=monkeypatch)
    assert term.tolist() == [False, True, True, True, True, True]
    assert rew.tolist() == [0.0, -10.0, -10.0, -10.0, -10.0, -10.0]


def test_box_ignores_goal_distance(monkeypatch) -> None:
    term, _ = _run({"box": BOX}, [[-0.01, -0.15, 0.145]], goal=[1.0, 1.0, 1.0], monkeypatch=monkeypatch)
    assert not term.any()


def test_max_distance_still_works(monkeypatch) -> None:
    term, _ = _run({"max_distance": 0.04}, [[0.0, 0.0, 0.03], [0.0, 0.0, 0.05]], monkeypatch=monkeypatch)
    assert term.tolist() == [False, True]


def test_needs_a_check() -> None:
    with pytest.raises(ValueError, match="max_distance"):
        cat.CubeAwayTermination({"reward_scale": -1.0})

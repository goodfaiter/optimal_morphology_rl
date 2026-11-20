"""goal_position_reward: distance over the configured axes (default x, y, z)."""

from types import SimpleNamespace

import pytest
import torch

from optimal_morphology_rl.modules.rewards import goal_position_reward as gpr


def _env(object_pos: torch.Tensor, goal_pos: torch.Tensor):
    container = SimpleNamespace(reward_object=SimpleNamespace(name="cube"))
    env = SimpleNamespace(module_manager=SimpleNamespace(container=container), info={"rewards": {}}, total_num_envs=object_pos.shape[0])
    return env, SimpleNamespace(pos_in_world=object_pos), (goal_pos, None)


@pytest.mark.parametrize("axes, expected_dist", [(None, 0.05), (["x", "y"], 0.0)])
def test_axes_select_the_distance(monkeypatch, axes, expected_dist) -> None:
    env, sensor, goal = _env(torch.tensor([[0.1, 0.2, 0.35]]), torch.tensor([[0.1, 0.2, 0.30]]))
    monkeypatch.setattr(gpr, "find_kinematic_sensor", lambda c, n: sensor)
    monkeypatch.setattr(gpr, "goal_in_env_frame", lambda c: goal)
    config = {"scale": 1.0, "normalize": 0.05} | ({"axes": axes} if axes else {})
    reward = gpr.GoalPositionReward(config).compute(env)
    assert torch.allclose(reward, torch.exp(torch.tensor([-((expected_dist / 0.05) ** 2)])), atol=1e-6)


def test_unknown_axis_rejected() -> None:
    with pytest.raises(ValueError, match="axes"):
        gpr.GoalPositionReward({"axes": ["x", "w"]})

"""The adapt_force 'force_goal' tracking reward."""

from types import SimpleNamespace

import pytest
import torch

from optimal_morphology_rl.envs.adapt_force.force_goal import ForceTrackingReward


def test_tracking_reward_sums_the_kernels_of_its_widths() -> None:
    container = SimpleNamespace(fingertip_force=torch.tensor([1.0, 1.2, 0.8]), force_goal=torch.tensor([1.0, 1.0, 1.0]))
    env = SimpleNamespace(module_manager=SimpleNamespace(container=container), total_num_envs=3, info={"rewards": {}})
    reward = ForceTrackingReward({"scale": 1.0, "normalize": [0.3, 0.1]}).compute(env)
    error = torch.tensor([0.0, 0.2, -0.2])
    assert torch.allclose(reward, torch.exp(-((error / 0.3) ** 2)) + torch.exp(-((error / 0.1) ** 2)))
    assert reward[0].item() == pytest.approx(2.0)  # one per width on target
    # with overshoot allowed, a force up to goal + tolerance is on target for both widths
    reward = ForceTrackingReward({"normalize": [0.3, 0.1], "allow_overshoot": True, "overshoot_tolerance": 0.5}).compute(env)
    assert reward[1].item() == pytest.approx(2.0) and reward[2].item() < 2.0


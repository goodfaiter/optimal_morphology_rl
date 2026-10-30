"""cube_angular_velocity_reward: capped (overshoot free) vs symmetric (overshoot penalized) speed reward."""

import math

import torch

from optimal_morphology_rl.envs.adapt_cube.rewards.cube_angular_velocity_reward import _cube_angular_velocity_reward_jit


def _raw(speeds, target, capped):
    _, raw = _cube_angular_velocity_reward_jit(torch.tensor(speeds), target, 0.5, 1.0, capped)
    return raw.tolist()


def test_capped_reward_does_not_penalize_overshoot() -> None:
    slow, at, fast = _raw([0.5, 1.0, 2.0], 1.0, True)
    assert at == fast == 1.0 and math.isclose(slow, math.exp(-1.0), rel_tol=1e-6)


def test_uncapped_reward_penalizes_overshoot_like_a_shortfall() -> None:
    slow, at, fast = _raw([0.5, 1.0, 1.5], 1.0, False)
    assert at == 1.0 and math.isclose(slow, fast, rel_tol=1e-6) and fast < 1.0


def test_negative_target_spins_the_other_way() -> None:
    along, against = _raw([-1.0, 1.0], -1.0, False)
    assert along == 1.0 and math.isclose(against, math.exp(-16.0), rel_tol=1e-5)
    assert _raw([-3.0], -1.0, True) == [1.0]

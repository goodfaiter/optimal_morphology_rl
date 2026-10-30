"""OrientationRate: windowed angular velocity from the orientation history."""

import math

import torch

from optimal_morphology_rl.envs.adapt_cube.rewards.cube_rotation_reward import OrientationRate


def _quat_about_y(angle: float) -> torch.Tensor:
    return torch.tensor([[0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2)]])


def test_orientation_rate_fast_spin_does_not_wrap_over_the_window() -> None:
    # -20 rad/s turns 4 rad (> pi) per 0.2 s window: the rate must stay -20, not wrap to a forward roll.
    dt, steps = 0.005, 40
    rate = OrientationRate(dt, window_steps=steps)
    for i in range(3 * steps):
        omega = rate(_quat_about_y(-20.0 * dt * i))
    assert torch.allclose(omega[0], torch.tensor([0.0, -20.0, 0.0]), atol=1e-2)


def test_orientation_rate_vibration_cancels_and_reset_restarts() -> None:
    dt, steps = 0.005, 40
    rate = OrientationRate(dt, window_steps=steps)
    for i in range(2 * steps):
        omega = rate(_quat_about_y(0.3 * (i % 2)))  # +-60 rad/s every step, no net rotation
    assert abs(omega[0, 1].item()) < 60.0 / steps + 1e-3

    rate.reset(torch.tensor([True]))
    assert torch.allclose(rate(_quat_about_y(2.0)), torch.zeros(1, 3))

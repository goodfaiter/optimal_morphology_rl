"""Durations given in seconds (independent of frame_skip): the shared conversion, the episode length and the force
goal's resampling."""

import math
from types import SimpleNamespace

import pytest

from optimal_morphology_rl.modules.base_module import seconds_to_steps
from optimal_morphology_rl.modules.terminations.progress_module import ProgressModule


def test_seconds_to_steps_exact_and_rounded() -> None:
    assert seconds_to_steps(0.02, 0.005, "x") == 4
    assert seconds_to_steps(0.02, 0.02, "x") == 1
    with pytest.raises(ValueError, match="whole number"):
        seconds_to_steps(0.01, 0.02, "observation.hist_period")
    assert seconds_to_steps(12.0, 0.03, "x", exact=False) == 400
    assert seconds_to_steps(0.001, 0.02, "x", exact=False) == 1  # at least one step


@pytest.mark.parametrize("frame_skip", [1, 4, 10])
def test_max_episode_sec_is_the_same_time_for_any_frame_skip(frame_skip) -> None:
    module = ProgressModule({"max_episode_sec": 12.0})
    container = SimpleNamespace(control_dt=0.005 * frame_skip, total_num_envs=1, device="cpu")
    container.get = lambda key, default=None: getattr(container, key, default)
    module.post_finalize(container)
    assert module.max_episode_length * container.control_dt == pytest.approx(12.0)


def test_max_episode_sec_and_length_exclude_each_other() -> None:
    with pytest.raises(ValueError, match="not both"):
        ProgressModule({"max_episode_sec": 12.0, "max_episode_length": 2400})


def test_force_goal_mean_hold_gives_the_same_hold_time_for_any_frame_skip() -> None:
    # The per-step probability 1 - exp(-dt / T): the mean hold (dt / p) stays ~T.
    for dt in (0.005, 0.02, 0.05):
        p = 1.0 - math.exp(-dt / 1.67)
        assert dt / p == pytest.approx(1.67, rel=0.02)

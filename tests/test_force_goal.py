"""The adapt_force 'force_goal' module: the ramp and the sinusoid's random center, the tracking reward and the
fingertip force observation."""

from types import SimpleNamespace

import pytest
import torch

from optimal_morphology_rl.envs.adapt_force.force_goal import ForceGoalModule, ForceTrackingReward

N = 256


def _goal(types, force_range=(0.5, 5.0)):
    module = ForceGoalModule({"force_range": list(force_range), "types": types})
    sensors = SimpleNamespace(force_sensor_link_names=["Middle_DP"], force_sensor_buf=torch.zeros(N, 1, 6))
    container = SimpleNamespace(total_num_envs=N, device=torch.device("cpu"), timestep=0.005, frame_skip=4,
                                force_sensors=sensors, reset_buf=torch.ones(N, dtype=torch.bool))
    container.get = lambda key, default=None: getattr(container, key, default)
    module.post_finalize(container)
    module.reset(container)
    return module, container


def test_ramp_moves_at_its_rate_and_holds_at_the_target() -> None:
    module, container = _goal({"ramp": {"ramp": {"min_rate": 0.5, "max_rate": 0.5, "mean_hold_sec": 0.0}}})
    start = container.force_goal.clone()
    target = module.ramp_target.clone()
    module.advance(container)
    moved = container.force_goal - start
    far = (target - start).abs() > 0.02  # not reached within one 20 ms step
    assert torch.allclose(moved[far].abs(), torch.full_like(moved[far], 0.5 * 0.02), atol=1e-6)  # 0.5 N/s
    assert torch.all(torch.sign(moved[far]) == torch.sign(target - start)[far])


def test_ramp_stays_in_the_range_and_reaches_its_targets() -> None:
    module, container = _goal({"ramp": {"ramp": {"min_rate": 0.1, "max_rate": 1.0, "mean_hold_sec": 0.5}}})
    held = torch.zeros(N, dtype=torch.bool)
    for _ in range(1000):  # 20 s
        module.advance(container)
        assert torch.all((container.force_goal >= 0.5 - 1e-6) & (container.force_goal <= 5.0 + 1e-6))
        held |= module.ramp_hold > 0.0
    assert held.float().mean() > 0.9  # most envs reached a target and held it


def test_sinusoid_random_center_keeps_the_whole_sine_in_the_range() -> None:
    cfg = {"min_frequency": 0.1, "max_frequency": 0.5, "min_amplitude_fraction": 0.2, "max_amplitude_fraction": 0.6,
           "random_center": True}
    module, container = _goal({"sinusoidal": {"sinusoidal": cfg}})
    assert torch.all(module.center - module.amp >= 0.5 - 1e-6) and torch.all(module.center + module.amp <= 5.0 + 1e-6)
    assert module.center.std() > 0.2  # not always the middle
    for _ in range(200):
        module.advance(container)
        assert torch.all((container.force_goal >= 0.5 - 1e-6) & (container.force_goal <= 5.0 + 1e-6))


def test_unknown_goal_type_raises() -> None:
    with pytest.raises(ValueError, match="unknown goal types"):
        ForceGoalModule({"types": {"square": {}}})


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


def _force_env(force):
    container = SimpleNamespace(fingertip_force=torch.tensor(force), reset_buf=torch.ones(len(force), dtype=torch.bool))
    container.get = lambda key, default=None: getattr(container, key, default)
    return SimpleNamespace(module_manager=SimpleNamespace(container=container))


def test_fingertip_force_without_noise_is_the_force() -> None:
    from optimal_morphology_rl.envs.adapt_force.force_goal import FingertipForceObservation

    env, out = _force_env([0.0, 1.5]), torch.zeros(2, 1)
    module = FingertipForceObservation()
    module.reset(env)
    module.compute_observation(env, out)
    assert torch.equal(out[:, 0], torch.tensor([0.0, 1.5]))


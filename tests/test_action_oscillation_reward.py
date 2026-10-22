"""Unit tests for the band-pass action oscillation penalty."""

import math
from types import SimpleNamespace

import pytest
import torch

from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.rewards.action_oscillation_reward import ActionOscillationReward


def _make(num_envs=2, **config):
    container = ModuleContainer()
    container.timestep, container.frame_skip = 0.0025, 2  # 200 Hz control
    container.actions = torch.zeros(num_envs, 2)
    module = ActionOscillationReward({"scale": 1.0, "low_hz": 2.0, "high_hz": 10.0, **config})
    module.post_finalize(container)
    env = SimpleNamespace(act_buf=torch.zeros(num_envs, 2), last_act_buf=torch.zeros(num_envs, 2), total_num_envs=num_envs,
                          info={"rewards": {}})
    return module, container, env


def _run(module, env, signal):
    """Feed a (T, num_envs, 2) action sequence, return the per-step rewards (T, num_envs)."""
    rewards = []
    for a in signal:
        env.last_act_buf, env.act_buf = env.act_buf.clone(), a.clone()
        rewards.append(module.compute(env))
    return torch.stack(rewards)


def _sine(hz, steps=600, amp=1.0):
    t = torch.arange(steps) * 0.005
    return (amp * torch.sin(2 * math.pi * hz * t))[:, None, None].expand(steps, 2, 2)


def test_constant_actions_are_not_penalized() -> None:
    module, _, env = _make()
    rewards = _run(module, env, torch.full((200, 2, 2), 0.7))
    assert torch.allclose(rewards[-1], torch.zeros(2), atol=1e-6)  # the step at t=0 decays away


def test_band_frequencies_are_penalized_more_than_slow_and_fast_ones() -> None:
    energy = {}
    for hz in (0.2, 5.0, 60.0):
        module, _, env = _make()
        energy[hz] = -_run(module, env, _sine(hz))[300:].mean().item()
    assert energy[5.0] > 5 * energy[0.2] and energy[5.0] > 5 * energy[60.0]


def test_reward_is_negative_scaled_and_logged() -> None:
    module, _, env = _make(scale=2.0)
    rewards = _run(module, env, _sine(5.0, steps=100))
    assert torch.all(rewards <= 0.0)
    assert env.info["rewards"]["action_oscillation_penalty"] == pytest.approx(rewards[-1].mean().item() / 2.0)


def test_reset_clears_the_filter_of_resetting_envs() -> None:
    module, container, env = _make()
    _run(module, env, _sine(5.0, steps=50))
    container.reset_buf = torch.tensor([True, False])
    module.reset(container)
    assert torch.all(module.band[0] == 0.0) and torch.all(module.high_pass[0] == 0.0)
    assert torch.any(module.band[1] != 0.0)


@pytest.mark.parametrize("config, match", [({"low_hz": 10.0, "high_hz": 2.0}, "low_hz"), ({"high_hz": 150.0}, "Nyquist")])
def test_invalid_config_raises(config, match) -> None:
    with pytest.raises(ValueError, match=match):
        _make(**config)

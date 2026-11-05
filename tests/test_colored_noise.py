"""Tests for colored (pink) noise generation and its PPO application."""

import numpy as np
import pytest
import torch

from optimal_morphology_rl.algos.colored_noise_agent import ColoredNoiseA2CAgent
from optimal_morphology_rl.helpers.colored_noise import powerlaw_psd_gaussian


def test_output_shape():
    out = powerlaw_psd_gaussian(1.0, (64, 4, 3), device="cpu")
    assert out.shape == (64, 4, 3)


def test_int_size():
    out = powerlaw_psd_gaussian(1.0, 64, device="cpu")
    assert out.shape == (64,)


def test_unit_std_per_series():
    out = powerlaw_psd_gaussian(1.0, (256, 8, 2), device="cpu")
    std = out.std(dim=0)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-4)


def test_zero_mean():
    out = powerlaw_psd_gaussian(1.0, (256, 8, 2), device="cpu")
    assert out.mean(dim=0).abs().max() < 1e-6


def test_seed_determinism():
    a = powerlaw_psd_gaussian(1.0, (64, 2, 2), device="cpu", generator=torch.Generator().manual_seed(7))
    b = powerlaw_psd_gaussian(1.0, (64, 2, 2), device="cpu", generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def _lag1_autocorr(x: torch.Tensor) -> float:
    x = x - x.mean(dim=0, keepdim=True)
    return ((x[:-1] * x[1:]).mean() / (x * x).mean()).item()


def test_beta_zero_is_white():
    white = powerlaw_psd_gaussian(0.0, (4096, 16, 4), device="cpu")
    assert abs(_lag1_autocorr(white)) < 0.05


def test_pink_noise_is_correlated():
    pink = powerlaw_psd_gaussian(1.0, (4096, 16, 4), device="cpu")
    assert _lag1_autocorr(pink) > 0.2


@pytest.mark.parametrize("beta", [0.0, 1.0, 2.0])
def test_spectral_slope(beta: float):
    seq_len = 8192
    out = powerlaw_psd_gaussian(beta, (seq_len, 32, 2), device="cpu")
    psd = torch.fft.rfft(out, dim=0).abs().pow(2).mean(dim=(1, 2))
    freqs = torch.fft.rfftfreq(seq_len)
    valid = (freqs > 0) & (freqs < 0.4)
    slope = np.polyfit(torch.log10(freqs[valid]), torch.log10(psd[valid]), 1)[0]
    assert slope == pytest.approx(-beta, abs=0.15)


def test_apply_colored_noise_matches_gaussian_logprob():
    torch.manual_seed(1)
    mu = torch.randn(5, 3)
    sigma = torch.rand(5, 3) + 0.5
    noise = torch.randn(5, 3)
    res = {"mus": mu.clone(), "sigmas": sigma.clone()}

    ColoredNoiseA2CAgent.apply_colored_noise(res, noise, scale=1.0)

    assert torch.allclose(res["actions"], mu + noise * sigma)
    ref = -torch.distributions.Normal(mu, sigma).log_prob(res["actions"]).sum(dim=-1)
    assert torch.allclose(res["neglogpacs"], ref, atol=1e-5)


def test_apply_colored_noise_scale():
    torch.manual_seed(2)
    mu = torch.randn(5, 3)
    sigma = torch.rand(5, 3) + 0.5
    noise = torch.randn(5, 3)
    res = {"mus": mu.clone(), "sigmas": sigma.clone()}

    ColoredNoiseA2CAgent.apply_colored_noise(res, noise, scale=0.5)

    assert torch.allclose(res["actions"], mu + noise * sigma * 0.5)


def test_apply_colored_noise_is_inplace():
    torch.manual_seed(3)
    noise = torch.randn(5, 3)
    res = {"mus": torch.zeros(5, 3), "sigmas": torch.ones(5, 3)}
    out = ColoredNoiseA2CAgent.apply_colored_noise(res, noise, scale=1.0)
    assert out is res
    assert res["actions"].shape == (5, 3)
    assert res["neglogpacs"].shape == (5,)


def test_continued_noise_starts_on_the_previous_sequence() -> None:
    torch.manual_seed(0)
    first, tail = ColoredNoiseA2CAgent.continued_noise(2.0, 256, 4, 2, 64, None)
    second, _ = ColoredNoiseA2CAgent.continued_noise(2.0, 256, 4, 2, 64, tail)
    assert first.shape == (256, 4, 2) and tail.shape == (64, 4, 2)
    torch.testing.assert_close(second[0], tail[0])  # the previous sequence's next sample


def test_continued_noise_has_no_jump_between_rollouts_and_keeps_its_level() -> None:
    torch.manual_seed(0)
    tail, rollouts = None, []
    for _ in range(40):
        rollout, tail = ColoredNoiseA2CAgent.continued_noise(2.0, 256, 256, 1, 64, tail)
        rollouts.append(rollout)
    noise = torch.cat(rollouts)  # (40 * 256, envs, 1)
    steps = (noise[1:] - noise[:-1]).abs()
    boundary = torch.stack([steps[k * 256 - 1] for k in range(1, 40)]).mean()
    assert boundary < 2.0 * steps.mean()  # independent rollouts jump by ~1.1 (std sqrt(2)) here
    blended = torch.cat([r[:64] for r in rollouts[1:]])
    assert abs(float(blended.std()) - 1.0) < 0.1  # the cos / sin blend keeps the unit std


def test_independent_rollouts_jump_at_the_boundary() -> None:
    torch.manual_seed(0)
    a = powerlaw_psd_gaussian(2.0, (256, 256, 1)); b = powerlaw_psd_gaussian(2.0, (256, 256, 1))
    within = (a[1:] - a[:-1]).abs().mean()
    assert (b[0] - a[-1]).abs().mean() > 5 * within

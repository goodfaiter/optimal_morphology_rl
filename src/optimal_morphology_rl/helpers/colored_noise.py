"""Temporally correlated colored-noise generation for action exploration."""

import torch


def powerlaw_psd_gaussian(beta, size, device=None, generator=None, time_dim=0):
    """Generate Gaussian noise with power spectral density proportional to 1/f^beta.

    White noise is filtered in the frequency domain with a f^(-beta/2) amplitude
    response and normalized so that every series along ``time_dim`` has unit
    standard deviation. ``beta=0`` gives white noise, ``beta=1`` pink noise and
    ``beta=2`` brown noise.

    Args:
        beta (:obj:`float`): Spectral slope exponent (PSD ~ 1/f^beta).
        size (int or tuple of int): Output shape; ``size[time_dim]`` is the
            series length.
        device: Torch device for the output tensor.
        generator (:obj:`torch.Generator`, optional): RNG used for the noise.
        time_dim (int): Dimension holding the time axis (default 0).

    Returns:
        :obj:`torch.Tensor`: Noise of shape ``size`` with unit std per series.
    """
    if isinstance(size, int):
        size = (size,)
    size = tuple(size)

    noise = torch.randn(size, device=device, generator=generator)
    seq_len = size[time_dim]
    if seq_len < 2 or beta == 0.0:
        return _unit_std(noise, time_dim)

    freqs = torch.fft.rfftfreq(seq_len, device=noise.device)
    amps = torch.zeros_like(freqs)
    nonzero = freqs > 0
    amps[nonzero] = freqs[nonzero].pow(-0.5 * beta)

    view = [1] * noise.dim()
    view[time_dim] = -1
    spectrum = torch.fft.rfft(noise, dim=time_dim) * amps.view(view)
    filtered = torch.fft.irfft(spectrum, n=seq_len, dim=time_dim)
    return _unit_std(filtered, time_dim)


def _unit_std(tensor, time_dim):
    std = tensor.std(dim=time_dim, keepdim=True).clamp_min(1e-12)
    return tensor / std

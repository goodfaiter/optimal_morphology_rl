"""Shared helpers for converting tendon config values to tensors."""

from __future__ import annotations

from typing import Any

import torch


def build_tendon_scale_tensor(cfg_value: Any, expected_len: int, device: torch.device, name: str) -> torch.Tensor:
    """Convert a scalar or per-tendon list config value to a float tensor."""
    if isinstance(cfg_value, (list, tuple)):
        if len(cfg_value) != expected_len:
            raise RuntimeError(f"Tendon config '{name}' length ({len(cfg_value)}) must match {expected_len}.")
        return torch.tensor(cfg_value, device=device, dtype=torch.float32)
    return torch.full((expected_len,), float(cfg_value), device=device, dtype=torch.float32)

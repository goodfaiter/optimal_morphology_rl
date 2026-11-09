"""Module that randomizes the 'antagonistic_spring' spring constants and preloads."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.base_module import BaseModule, curriculum_range
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("randomize_antagonistic_springs")
class RandomizeAntagonisticSpringsModule(BaseModule):
    """Randomizes the 'antagonistic_spring' constants on reset, per env and motor.

    Each constant is resampled for the resetting envs as ``nominal * U[1 - scale, 1 + scale]``,
    independently per env and motor (``scale: 0.1`` = within 10 % of the configured
    ``spring_constants``; motors without a spring stay at 0). The width follows the training curriculum
    (``scale * curriculum_factor``: the nominal constants at 0, the full range at 1).

    With ``preload_range`` ([low, high], Nm) the preload (pretension) of every motor with a spring is drawn uniformly
    from that absolute range, per env and motor on reset (the nominal preload may be 0, so a relative scale would not
    work); the curriculum shrinks the range toward the configured ``spring_preloads``.

    Config shape::

        randomize_antagonistic_springs:
          scale: 0.1
          preload_range: [0.0, 0.01] # Nm, optional
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.scale = float(self.config.get("scale", 0.1))
        if not 0.0 <= self.scale <= 1.0:
            raise RuntimeError(f"RandomizeAntagonisticSprings config 'scale' must be in [0, 1], got {self.scale}.")
        preload_range = self.config.get("preload_range")
        self.preload_range = None if preload_range is None else (float(preload_range[0]), float(preload_range[1]))
        if self.preload_range is not None and not 0.0 <= self.preload_range[0] <= self.preload_range[1]:
            raise RuntimeError(f"RandomizeAntagonisticSprings 'preload_range' must be [low, high] with 0 <= low <= high, got {preload_range}.")
        self.spring = None

    def _setup(self, container: ModuleContainer) -> None:
        manager = container.env.module_manager
        if "antagonistic_spring" not in manager:
            raise RuntimeError("RandomizeAntagonisticSprings requires the 'antagonistic_spring' module.")
        self.spring = manager.get("antagonistic_spring")

    def reset(self, container: ModuleContainer) -> None:
        """Resample the spring constants of the resetting envs."""
        if self.spring is None:
            self._setup(container)
        reset_buf = container.reset_buf
        if not reset_buf.any():
            return
        values = self.spring.spring_constants_buf
        factor = 1.0 + self.scale * self.curriculum_factor * (2.0 * torch.rand_like(values) - 1.0)
        values[reset_buf] = (self.spring.spring_constants * factor)[reset_buf]
        if self.preload_range is not None:
            nominal = self.spring.spring_preloads
            low, high = curriculum_range(*self.preload_range, self.curriculum_factor, nominal=nominal)
            preloads = (low + (high - low) * torch.rand_like(values)).clamp(min=0.0)
            preloads = preloads * (self.spring.spring_constants > 0.0)  # only motors with a spring
            self.spring.spring_preloads_buf[reset_buf] = preloads[reset_buf]

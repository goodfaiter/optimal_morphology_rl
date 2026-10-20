"""Module that randomizes the 'antagonistic_spring' spring constants."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("randomize_antagonistic_springs")
class RandomizeAntagonisticSpringsModule(BaseModule):
    """Randomizes the 'antagonistic_spring' constants on reset, per env and motor.

    Each constant is resampled for the resetting envs as ``nominal * U[1 - scale, 1 + scale]``,
    independently per env and motor (``scale: 0.1`` = within 10 % of the configured
    ``spring_constants``; motors without a spring stay at 0).

    Config shape::

        randomize_antagonistic_springs:
          scale: 0.1
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.scale = float(self.config.get("scale", 0.1))
        if not 0.0 <= self.scale <= 1.0:
            raise RuntimeError(f"RandomizeAntagonisticSprings config 'scale' must be in [0, 1], got {self.scale}.")
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
        factor = 1.0 + self.scale * (2.0 * torch.rand_like(values) - 1.0)
        values[reset_buf] = (self.spring.spring_constants * factor)[reset_buf]

"""Module that randomizes the friction of 'apply_m5_forces' per env."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("randomize_m5")
class RandomizeM5Module(BaseModule):
    """Redraws the friction scale of 'apply_m5_forces' (``friction_scale``, one factor per env) on reset.

    For the resetting envs the factor is drawn from ``U[1 - scale, 1 + scale]`` (``scale: 0.25`` = +-25 %) and
    multiplies tau_f_max (the M5 budget, the learned envelope, or the learned friction torque) until the env's
    next reset. The width follows the training curriculum (``scale * curriculum_factor``: 1 at 0, the full range
    at 1).

    Config shape::

        randomize_m5:
          scale: 0.25
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.scale = float(self.config.get("scale", 0.1))
        if not 0.0 <= self.scale <= 1.0:
            raise RuntimeError(f"RandomizeM5Module config 'scale' must be in [0, 1], got {self.scale}.")
        self.m5 = None

    def reset(self, container: ModuleContainer) -> None:
        """Redraw the friction scale of the resetting envs."""
        if self.m5 is None:
            manager = container.env.module_manager
            if "apply_m5_forces" not in manager:
                raise RuntimeError("RandomizeM5Module requires the 'apply_m5_forces' module.")
            self.m5 = manager.get("apply_m5_forces")
        reset_buf = container.reset_buf
        if not reset_buf.any():
            return
        scale = self.m5.friction_scale
        factor = 1.0 + self.scale * self.curriculum_factor * (2.0 * torch.rand_like(scale) - 1.0)
        scale[reset_buf] = factor[reset_buf]

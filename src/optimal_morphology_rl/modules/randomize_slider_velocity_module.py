"""Module that randomizes the servo speed limit ('max_slider_velocity') of 'apply_m5_forces'."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.base_module import BaseModule, curriculum_range
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("randomize_slider_velocity")
class RandomizeSliderVelocityModule(BaseModule):
    """Randomizes the slider speed limit of 'apply_m5_forces' on reset, per env and tendon.

    The real servo is voltage limited (PWM saturates at about 7.6 rad/s on the spool, lower under
    load), so its top speed varies. For the resetting envs ``max_slider_velocity`` (m/s at the
    spool rim) is resampled uniformly from ``max_slider_velocity_range``. Requires
    'apply_m5_forces' with a configured ``max_slider_velocity``. The range follows the training curriculum
    from that configured value (at 0) to ``max_slider_velocity_range`` (at 1).

    Config shape::

        randomize_slider_velocity:
          max_slider_velocity_range: [0.032, 0.040]  # m/s
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        value = self.config.get("max_slider_velocity_range")
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise RuntimeError(f"RandomizeSliderVelocityModule config 'max_slider_velocity_range' must be a [min, max] list, got {value}.")
        self.low, self.high = float(value[0]), float(value[1])
        if not 0.0 < self.low <= self.high:
            raise RuntimeError(f"RandomizeSliderVelocityModule 'max_slider_velocity_range' must satisfy 0 < min <= max, got {value}.")
        self.m5 = None

    def _setup(self, container: ModuleContainer) -> None:
        manager = container.env.module_manager
        if "apply_m5_forces" not in manager:
            raise RuntimeError("RandomizeSliderVelocityModule requires the 'apply_m5_forces' module.")
        self.m5 = manager.get("apply_m5_forces")
        if self.m5.max_slider_velocity is None:
            raise RuntimeError("RandomizeSliderVelocityModule requires 'apply_m5_forces' to set 'max_slider_velocity'.")
        self.nominal = self.m5.max_slider_velocity.clone()

    def reset(self, container: ModuleContainer) -> None:
        """Resample the speed limit of the resetting envs."""
        if self.m5 is None:
            self._setup(container)
        reset_buf = container.reset_buf
        if not reset_buf.any():
            return
        values = self.m5.max_slider_velocity
        low, high = curriculum_range(self.low, self.high, self.curriculum_factor, nominal=self.nominal)
        samples = low + (high - low) * torch.rand_like(values)
        values[reset_buf] = samples[reset_buf]

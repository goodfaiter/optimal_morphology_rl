"""Module that randomizes the 'apply_m5_forces' servo law parameters (M5 friction coefficients)."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.apply_control.apply_m5_forces_module import M5_PARAMS, MODEL_COEFFICIENTS
from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module

DEFAULT_PARAMS = ["K_v", "K_c", "K_m", "K_e", "v_s", "alpha", "K_cs", "K_ms", "K_es"]


@register_module("randomize_m5")
class RandomizeM5Module(BaseModule):
    """Randomizes the M5 servo parameters of 'apply_m5_forces' on reset, per env and tendon.

    Every parameter in ``params`` is resampled for the resetting envs as
    ``nominal * U[1 - scale, 1 + scale]``, independently per env, tendon and parameter
    (``scale: 0.1`` = within 10 %). ``params`` defaults to the M5 friction coefficients
    (K_v, K_c, K_m, K_e, v_s, alpha, K_cs, K_ms, K_es); ``motor_gain`` can be added.

    With a learned friction model ('friction_model_path') the coefficients come from the model: then
    the same ``scale`` and ``params`` apply to the model's predictions. Each predicted coefficient
    (K_c, K_v, K_m, K_e, K_cs, K_ms, K_es) listed in ``params`` is scaled by a factor in
    ``U[1 - scale, 1 + scale]`` per env and tendon (``apply_m5_forces.model_coefficient_scale``),
    ``motor_gain`` is randomized as without a model, and v_s / alpha (fixed by the model) and the
    configured constants are unused.

    Config shape::

        randomize_m5:
          scale: 0.1
          params: [K_v, K_c, K_m, K_e, v_s, alpha, K_cs, K_ms, K_es]  # optional
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.scale = float(self.config.get("scale", 0.1))
        if not 0.0 <= self.scale < 1.0:
            raise RuntimeError(f"RandomizeM5Module config 'scale' must be in [0, 1), got {self.scale}.")
        self.params = list(self.config.get("params", DEFAULT_PARAMS))
        unknown = [name for name in self.params if name not in M5_PARAMS]
        if unknown:
            raise RuntimeError(f"RandomizeM5Module 'params' {unknown} are not servo parameters {list(M5_PARAMS)}.")
        self.model_params = []
        self.m5 = None

    def _setup(self, container: ModuleContainer) -> None:
        manager = container.env.module_manager
        if "apply_m5_forces" not in manager:
            raise RuntimeError("RandomizeM5Module requires the 'apply_m5_forces' module.")
        self.m5 = manager.get("apply_m5_forces")
        if self.m5.friction_model is not None:
            # The constants other than motor_gain are not used with a friction model: randomize its output instead.
            self.model_params = [MODEL_COEFFICIENTS.index(name) for name in self.params if name in MODEL_COEFFICIENTS]
            self.params = [name for name in self.params if name == "motor_gain"]

    def reset(self, container: ModuleContainer) -> None:
        """Resample the parameters of the resetting envs."""
        if self.m5 is None:
            self._setup(container)
        reset_buf = container.reset_buf
        if not reset_buf.any():
            return
        for name in self.params:
            values = getattr(self.m5, M5_PARAMS[name])
            factor = 1.0 + self.scale * (2.0 * torch.rand_like(values) - 1.0)
            values[reset_buf] = (self.m5.nominal[name] * factor)[reset_buf]
        if self.model_params:
            model_scale = self.m5.model_coefficient_scale
            factor = torch.ones_like(model_scale)
            draws = factor[..., self.model_params]
            factor[..., self.model_params] = 1.0 + self.scale * (2.0 * torch.rand_like(draws) - 1.0)
            model_scale[reset_buf] = factor[reset_buf]

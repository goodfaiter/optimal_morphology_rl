"""Manager module that owns observation sub-modules and builds the obs vector."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from vlearn.spaces import Box

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import ModuleManager, register_module
from optimal_morphology_rl.modules.observations.observation_base_module import (
    ObservationBaseModule,
)


# Local registry for observation sub-modules.
OBSERVATION_REGISTRY: dict[str, type[ObservationBaseModule]] = {}


def register_observation(name: str | None = None):
    """Decorator that registers an observation sub-module class."""

    def decorator(cls: type[ObservationBaseModule]) -> type[ObservationBaseModule]:
        registry_name = name if name is not None else cls.__name__
        if registry_name in OBSERVATION_REGISTRY:
            raise ValueError(f"Observation module '{registry_name}' is already registered.")
        OBSERVATION_REGISTRY[registry_name] = cls
        return cls

    return decorator


@register_module("observation")
class ObservationManagerModule(BaseModule):
    """Builds the observation vector from registered observation sub-modules.

    Config shape::

        observation:
          num_hist: 3
          hist_stride: 10
          modules:
            - robot_state
            - object_state
            - goal_state

    The sub-module configs live at the same level as ``modules`` inside the
    ``observation`` block.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.num_hist = int(self.config.get("num_hist", 1))
        self.hist_stride = int(self.config.get("hist_stride", 1))

        self.sub_manager = self._build_sub_manager()
        self._obs_modules: list[ObservationBaseModule] = [m for m in self.sub_manager if isinstance(m, ObservationBaseModule)]

        self.base_obs: torch.Tensor | None = None
        self.obs_history: Any | None = None
        self._obs_slices: dict[str, slice] | None = None
        self._base_obs_dim: int | None = None

    def _build_sub_manager(self) -> ModuleManager:
        """Create a ModuleManager for observation sub-modules."""
        sub_config: dict[str, Any] = {"modules": {"init_modules": []}}
        for key in self.config.keys():
            if key.startswith("_"):
                continue
            if key == "modules":
                sub_config["modules"]["init_modules"] = list(self.config[key])
            else:
                sub_config[key] = self.config[key]
        return ModuleManager.from_config(sub_config, registry=OBSERVATION_REGISTRY)

    def finalize(self, container: ModuleContainer) -> None:
        """Compute obs dims, build slices, and set the env observation space."""
        env = container.get("env")
        if env is None:
            raise RuntimeError("ObservationManagerModule requires 'env' in the shared container.")

        offset = 0
        self._obs_slices = {}

        for module in self._obs_modules:
            name = type(module).__name__
            dim = module.get_obs_dim(env)
            self._obs_slices[name] = slice(offset, offset + dim)
            offset += dim

        self._base_obs_dim = offset
        num_obs = self._base_obs_dim * self.num_hist

        env.observation_space = Box(
            low=np.full(num_obs, np.finfo("f").min, dtype=np.float32),
            high=np.full(num_obs, np.finfo("f").max, dtype=np.float32),
            dtype=np.float32,
        )

        print(f"Observation space size: {num_obs} (base={self._base_obs_dim}, num_hist={self.num_hist}, stride={self.hist_stride})")

    def post_finalize(self, container: ModuleContainer) -> None:
        """Allocate observation buffer, base obs, and history buffers."""
        env = container.env
        device = env.device
        total_num_envs = env.total_num_envs

        container.obs_buf = torch.zeros(
            (total_num_envs,) + env.observation_space.shape,
            device=device,
            dtype=torch.float32,
        )

        self.base_obs = torch.zeros((total_num_envs, self._base_obs_dim), device=device, dtype=torch.float32)

        obs_history_length = 1 + (self.num_hist - 1) * self.hist_stride
        # Lazy import to avoid a hard dependency if history is not used.
        from time_series_buffer.time_series_buffer import TimeSeriesBuffer

        self.obs_history = TimeSeriesBuffer(
            num_envs=total_num_envs,
            dim=self._base_obs_dim,
            max_size=obs_history_length,
            stride=self.hist_stride,
            device=device,
        )

    def step(self, container: ModuleContainer) -> None:
        """Fill ``env.obs_buf`` from sub-module observations and history."""
        env = container.get("env")
        if env is None:
            raise RuntimeError("ObservationManagerModule requires 'env' in the shared container.")
        if self.base_obs is None or self._obs_slices is None:
            raise RuntimeError("ObservationManagerModule has not been finalized.")

        for module in self._obs_modules:
            name = type(module).__name__
            slc = self._obs_slices[name]
            module.compute_observation(env, self.base_obs[:, slc])

        self.obs_history.add(self.base_obs)
        env.obs_buf[:] = self.obs_history.get().view(env.total_num_envs, -1)

    def reset(self, container: ModuleContainer) -> None:
        """Reset the observation history."""
        reset_buf = container.get("reset_buf")
        if self.obs_history is not None and reset_buf is not None:
            self.obs_history.reset_idx(reset_buf)

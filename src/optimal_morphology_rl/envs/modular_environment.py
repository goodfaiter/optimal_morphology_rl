"""Fresh, module-driven environment that does not inherit from EnvironmentGpu."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import vlearn as v
import yaml
from vlearn.spaces import Box

from optimal_morphology_rl.modules.module_manager import ModuleManager


class ModularEnvironment:
    """RL environment assembled from YAML-configured modules.

    This class is intentionally independent of ``EnvironmentGpu``.  It owns the
    standard RL buffers and orchestrates module lifecycle hooks.

    Args:
        config: Path to a YAML file or a configuration dictionary.
        registry: Optional module registry.  Uses the global default if ``None``.
    """

    def __init__(
        self,
        config: str | Path | dict[str, Any],
        registry: dict[str, type] | None = None,
    ):
        self.raw_config = self._load_config(config)

        # Spaces (populated by modules during finalize).
        self._observation_space: Box | None = None
        self._action_space: Box | None = None
        self._info: dict[str, Any] = {"extras": {}, "rewards": {}}

        # Build modules and run lifecycle hooks.
        self.module_manager = ModuleManager.from_config(self.raw_config, registry=registry)
        self.module_manager.container.env = self
        self.module_manager.finalize()

        # Post-finalize allocates buffers (obs/rew/act) and builds GPU commands.
        self.module_manager.post_finalize()

        # Initial reset to populate observations.
        self.reset()

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------
    @staticmethod
    def _load_config(config: str | Path | dict[str, Any]) -> dict[str, Any]:
        if isinstance(config, (str, Path)):
            with open(config, "r") as f:
                loaded = yaml.safe_load(f)
            if loaded is None:
                loaded = {}
            return loaded
        return dict(config)

    # ------------------------------------------------------------------
    # Spaces
    # ------------------------------------------------------------------
    @property
    def observation_space(self) -> Box:
        if self._observation_space is None:
            raise AttributeError("observation_space has not been set")
        return self._observation_space

    @observation_space.setter
    def observation_space(self, space: Box) -> None:
        if not isinstance(space, Box):
            raise TypeError("observation_space must be a vlearn.spaces.Box")
        self._observation_space = space

    @property
    def action_space(self) -> Box:
        if self._action_space is None:
            raise AttributeError("action_space has not been set")
        return self._action_space

    @action_space.setter
    def action_space(self, space: Box) -> None:
        if not isinstance(space, Box):
            raise TypeError("action_space must be a vlearn.spaces.Box")
        self._action_space = space

    # ------------------------------------------------------------------
    # Buffers (allocated by their owning modules during post_finalize)
    # ------------------------------------------------------------------
    @property
    def obs_buf(self) -> torch.Tensor:
        return self.module_manager.container.obs_buf

    @property
    def rew_buf(self) -> torch.Tensor:
        return self.module_manager.container.rew_buf

    @property
    def act_buf(self) -> torch.Tensor:
        return self.module_manager.container.act_buf

    @property
    def info(self) -> dict[str, Any]:
        return self._info

    # Core simulation parameters are owned by modules and shared via the container.
    @property
    def device(self) -> torch.device:
        return self.module_manager.container.device

    @property
    def num_envs(self) -> list[int]:
        return self.module_manager.container.num_envs

    @property
    def total_num_envs(self) -> int:
        return self.module_manager.container.total_num_envs

    @property
    def frame_skip(self) -> int:
        return self.module_manager.container.frame_skip

    @property
    def rendering(self) -> bool:
        return self.module_manager.container.rendering

    @property
    def term_buf(self) -> torch.Tensor:
        return self.module_manager.container.term_buf

    @property
    def trunc_buf(self) -> torch.Tensor:
        return self.module_manager.container.trunc_buf

    @property
    def progress_buf(self) -> torch.Tensor:
        return self.module_manager.container.progress_buf

    @property
    def reset_buf(self) -> torch.Tensor:
        return self.module_manager.container.reset_buf

    # ------------------------------------------------------------------
    # Gym access
    # ------------------------------------------------------------------
    @property
    def gym(self) -> v.Gym:
        return self.module_manager.container.gym

    @property
    def env_group(self):
        return self.module_manager.container.env_group

    @property
    def env_sets(self):
        return self.module_manager.container.env_sets

    @property
    def env_def(self):
        return self.module_manager.container.env_def

    @property
    def env_def_handle(self):
        return self.module_manager.container.env_def_handle

    @property
    def gym_render(self):
        return self.module_manager.container.get("gym_render", None)

    # ------------------------------------------------------------------
    # RL interface
    # ------------------------------------------------------------------
    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Step the physics and return obs, reward, term, trunc, info."""
        if actions.device != self.device:
            actions = actions.to(self.device)
        self.act_buf[:] = actions

        self.module_manager.pre_physics_step()

        for _ in range(self.frame_skip):
            self.module_manager.pre_gym_step()
            self.gym.step()
            self.module_manager.post_gym_step()

        # Reset loop: envs that terminated/truncated in the previous step are reset.
        if self.reset_buf.any():
            self.module_manager.reset()

        # Refresh state and compute observations/rewards/terminations.
        self.module_manager.post_physics_step()

        return (
            self.obs_buf.clone(),
            self.rew_buf.clone(),
            self.term_buf.clone(),
            self.trunc_buf.clone(),
            self.info,
        )   

    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset all environments and return initial observations."""
        self.reset_buf[:] = True
        self.module_manager.reset()
        self.gym.step()
        self.gym.compute_kinematics()
        self.module_manager.post_physics_step()
        return self.obs_buf.clone(), {}

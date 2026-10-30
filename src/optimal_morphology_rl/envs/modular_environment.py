"""Fresh, module-driven environment that does not inherit from EnvironmentGpu."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import vlearn as v
from vlearn.spaces import Box

from optimal_morphology_rl.modules.module_manager import ModuleManager
from optimal_morphology_rl.utils.config import load_yaml_with_context, resolve_config


class ModularEnvironment:
    """RL environment assembled from YAML-configured modules.

    This class is intentionally independent of ``EnvironmentGpu``.  It owns the
    standard RL buffers and orchestrates module lifecycle hooks.

    Args:
        config: Path to a YAML file or a configuration dictionary.
        registry: Optional module registry.  Uses the global default if ``None``.
        context: Optional runtime variables exposed to OmegaConf interpolations
            and expressions (e.g. ``{"mode": "train"}``).
    """

    def __init__(
        self,
        config: str | Path | dict[str, Any],
        registry: dict[str, type] | None = None,
        context: dict[str, Any] | None = None,
    ):
        self.raw_config = self._load_config(config, context=context)

        # Spaces (populated by modules during finalize).
        self._observation_space: Box | None = None
        self._action_space: Box | None = None
        self._info: dict[str, Any] = {"extras": {}, "rewards": {}}

        # Build modules and run lifecycle hooks.
        self._check_post_physics_order(self.raw_config)
        self.module_manager = ModuleManager.from_config(self.raw_config, registry=registry)
        self.module_manager.container.env = self
        self.module_manager.finalize()

        # Post-finalize allocates buffers (obs/rew/act) and builds GPU commands.
        self.module_manager.post_finalize()

        # The training curriculum's first level (set by the runner from the ppo config), so the first episodes use
        # it: rl_games hands the factor over only from its first epoch on, after the initial resets, and the modules'
        # default 1.0 would fully randomize every env's first episode.
        initial_factor = self.raw_config.get("_initial_curriculum_factor")
        if initial_factor is not None:
            self.module_manager.set_curriculum_factor(float(initial_factor))

        # Initial reset to populate observations.
        self.reset()

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------
    @staticmethod
    def _load_config(
        config: str | Path | dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if isinstance(config, (str, Path)):
            return load_yaml_with_context(config, context=context)
        if context is not None:
            return resolve_config(config, context=context)
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
    def last_act_buf(self) -> torch.Tensor:
        return self.module_manager.container.last_act_buf

    @property
    def scaled_act_buf(self) -> torch.Tensor:
        return self.module_manager.container.scaled_act_buf

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
    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Step the physics and return obs, reward, term, trunc, info.

        Episodes that end in this step restart in it (gym's autoreset, as IsaacLab): the reward and the term / trunc
        flags are those of the state the physics reached, the observation is the new episode's first one. rl_games
        does not bootstrap from the observation returned with done (its GAE masks V(obs[t+1]) with dones[t+1]), so
        the terminal transition carries the termination reward and nothing crosses into the next episode.
        """
        self.module_manager.container.actions[:] = actions

        self.module_manager.pre_physics_step()

        for _ in range(self.frame_skip):
            self.module_manager.pre_gym_step()
            self.gym.step()
            self.module_manager.post_gym_step()
        # Kinematic sensor states are only up to date after forward kinematics (vlearn: call compute_kinematics()
        # after Gym.step()); without it the cube sensor's linear velocity read ~0.76x the object's. Costs ~nothing.
        self.gym.compute_kinematics()

        # Block the renderer from advancing once the frame-skip block is done;
        # this mirrors EnvironmentGpu.step and keeps rendering on the control
        # timestep rather than the physics timestep.
        if self.rendering and self.gym_render is not None:
            self.gym_render.set_step(False)

        # Rewards and terminations of the state the physics reached, the restart of the ended episodes (reset_envs)
        # and the observations (the new episodes' first for the restarted envs).
        self.module_manager.post_physics_step()
        container = self.module_manager.container
        term, trunc = container.episode_end_term, container.episode_end_trunc

        return (
            self._observations(),
            self.rew_buf.clone(),
            term,
            trunc,
            self.info,
        )

    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset all environments and return initial observations: every module's reset fills its part (the readers,
        e.g. update_robot, read the reset state), then the first observation."""
        container = self.module_manager.container
        self.reset_buf[:] = True
        self.module_manager.reset()
        self.module_manager.get("observation").step(container)
        return self._observations(), {}

    @staticmethod
    def _check_post_physics_order(config: dict[str, Any]) -> None:
        """Episodes restart in reset_envs: it must follow termination and precede observation (see ResetEnvsModule)."""
        names = list((config.get("modules") or {}).get("post_physics_step_modules") or [])
        names = [ModuleManager._extract_module_name(entry) for entry in names]
        if "termination" not in names and "observation" not in names:
            return
        if "reset_envs" not in names:
            raise ValueError("post_physics_step_modules has no 'reset_envs': ended episodes would never restart. List it "
                             "after 'termination' and before 'observation' (see ResetEnvsModule).")
        reset = names.index("reset_envs")
        if "termination" in names and names.index("termination") > reset:
            raise ValueError("post_physics_step_modules: 'reset_envs' must come after 'termination'.")
        if "observation" in names and (names.index("observation") < reset or names.count("observation") > 1):
            raise ValueError("post_physics_step_modules: 'observation' must come once, after 'reset_envs' (its history "
                             "takes one frame per call).")

    def _observations(self) -> torch.Tensor | dict[str, torch.Tensor]:
        """The policy observations, or {"obs", "priv_obs"} with privileged (critic only) observations."""
        priv_obs = self.module_manager.container.get("priv_obs_buf")
        if priv_obs is None:
            return self.obs_buf.clone()
        return {"obs": self.obs_buf.clone(), "priv_obs": priv_obs.clone()}

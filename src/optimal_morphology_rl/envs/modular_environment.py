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

        # Core simulation parameters are owned by the environment so that modules
        # and legacy helpers can read them directly.
        sim_config = self.raw_config.get("create_rigid_vsim_envs", {})
        self._num_envs = int(sim_config["num_envs"])
        self.device = torch.device(sim_config["device"])
        self.total_num_envs = self._num_envs
        # Several legacy helpers (Contacts, CameraRecorder) expect a list of
        # environment counts per set; a single set is represented by one entry.
        self.num_envs: list[int] = [self._num_envs]

        self.max_episode_length = int(
            sim_config.get("max_episode_length", 6 * 60)
        )
        self.timestep = float(sim_config.get("timestep", 1.0 / 120.0))
        self.frame_skip = int(sim_config.get("frame_skip", 2))
        self.dt = self.timestep * self.frame_skip
        self.spacing = float(sim_config.get("spacing", 0.5))
        self.rendering = bool(sim_config.get("rendering", False))
        self.with_window = bool(sim_config.get("with_window", True))
        self.max_contact_pairs_per_env = int(
            sim_config.get("max_contact_pairs_per_env", 128)
        )
        self.max_contact_patches_per_env = int(
            sim_config.get("max_contact_patches_per_env", self.max_contact_pairs_per_env)
        )
        self.max_contact_points_per_patch = int(
            sim_config.get("max_contact_points_per_patch", 4)
        )

        # Spaces (populated by modules during finalize).
        self._observation_space: Box | None = None
        self._action_space: Box | None = None
        self._render_finished: bool = False

        # Build modules and run lifecycle hooks.
        self.module_manager = ModuleManager.from_config(self.raw_config, registry=registry)
        self.module_manager.finalize(self)

        # Allocate standard RL buffers now that spaces are known.
        self.allocate_buffers()

        # Convenience aliases for legacy helpers (Contacts, etc.) must be
        # available before post_finalize because several modules use them.
        self._sync_legacy_aliases()

        # Post-finalize may use the standard buffers (e.g. robot_control).
        self.module_manager.post_finalize(self)

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
    # Buffer allocation
    # ------------------------------------------------------------------
    def allocate_buffers(self) -> None:
        """Allocate standard RL buffers."""
        self._obs_buf = torch.zeros(
            (self.total_num_envs,) + self.observation_space.shape,
            dtype=torch.float32,
            device=self.device,
        )
        self._rew_buf = torch.zeros(
            self.total_num_envs, dtype=torch.float32, device=self.device
        )
        self._term_buf = torch.zeros(
            self.total_num_envs, dtype=torch.bool, device=self.device
        )
        self._trunc_buf = torch.zeros(
            self.total_num_envs, dtype=torch.bool, device=self.device
        )
        self._progress_buf = torch.zeros(
            self.total_num_envs, dtype=torch.long, device=self.device
        )
        self._act_buf = torch.zeros(
            (self.total_num_envs,) + self.action_space.shape,
            device=self.device,
            dtype=torch.float32,
        )
        self._reset_buf = torch.ones(
            self.total_num_envs, dtype=torch.bool, device=self.device
        )
        self._info: dict[str, Any] = {"extras": {}, "rewards": {}}

    @property
    def obs_buf(self) -> torch.Tensor:
        return self._obs_buf

    @property
    def rew_buf(self) -> torch.Tensor:
        return self._rew_buf

    @property
    def term_buf(self) -> torch.Tensor:
        return self._term_buf

    @property
    def trunc_buf(self) -> torch.Tensor:
        return self._trunc_buf

    @property
    def progress_buf(self) -> torch.Tensor:
        return self._progress_buf

    @property
    def act_buf(self) -> torch.Tensor:
        return self._act_buf

    @property
    def reset_buf(self) -> torch.Tensor:
        return self._reset_buf

    @property
    def info(self) -> dict[str, Any]:
        return self._info

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
    # Aliases for legacy helpers
    # ------------------------------------------------------------------
    def _sync_legacy_aliases(self) -> None:
        """Mirror key container state on ``self`` for helpers like ``Contacts``."""
        container = self.module_manager.container
        self.robot = container.get("robot")
        self.objects = container.get("object_generator")

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

        self.module_manager.pre_physics_step(self)

        for _ in range(self.frame_skip):
            self.module_manager.pre_gym_step(self)
            self.gym.step()
            self.module_manager.post_gym_step(self)

        self.progress_buf[:] += 1
        self.module_manager.post_physics_step(self)

        # Reset loop: envs that terminated/truncated in the previous step are reset.
        self.reset_buf[:] = torch.logical_or(self.term_buf, self.trunc_buf)
        if self.reset_buf.any():
            self.reset_idx()
            self.module_manager.reset(self)

        # Refresh state after any resets and compute observations/rewards/terminations.
        self._refresh_state()
        self.observation_manager.compute_observations(self)
        self.reward_manager.compute(self)
        self.termination_manager.compute(self)

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
        self.reset_idx()
        self.module_manager.reset(self)
        self.gym.compute_kinematics()
        self._refresh_state()
        self.observation_manager.compute_observations(self)
        if self.total_num_envs != 1:
            self.progress_buf[:] = torch.randint(
                0, self.max_episode_length, (self.total_num_envs,), device=self.device
            )
        return self.obs_buf.clone(), {}

    def reset_idx(self) -> None:
        """Reset environment-owned buffers for the indices in ``reset_buf``."""
        if self.reset_buf.sum() == 0:
            return
        self.act_buf[self.reset_buf, :] = 0.0
        self.progress_buf[self.reset_buf] = 0
        self.term_buf[self.reset_buf] = False
        self.trunc_buf[self.reset_buf] = False

    def _refresh_state(self) -> None:
        """Refresh simulation state from all relevant modules."""
        container = self.module_manager.container

        if container.get("robot") is not None:
            container.robot.refresh_buffers(container.gym)

        if container.get("objects") is not None:
            for obj in container.objects.values():
                obj.refresh_buffers(container.gym)

        if container.get("kinematic_sensor") is not None:
            container.kinematic_sensor.update(container.gym)

        if container.get("force_sensors") is not None:
            container.force_sensors.update(container.gym)

    # ------------------------------------------------------------------
    # Manager accessors
    # ------------------------------------------------------------------
    @property
    def observation_manager(self):
        return self.module_manager["observation"]

    @property
    def reward_manager(self):
        return self.module_manager["reward"]

    @property
    def termination_manager(self):
        return self.module_manager["termination"]

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self) -> bool:
        """Render the environment if rendering is enabled."""
        if not self.rendering or self.gym_render is None:
            return False
        self._render_finished = self.gym_render.render(self.render_callback)
        return self._render_finished

    def render_callback(self) -> None:
        """Callback invoked by the renderer each frame."""
        pass

    @property
    def render_finished(self) -> bool:
        """Whether the renderer has finished."""
        return self._render_finished

    @render_finished.setter
    def render_finished(self, value: bool) -> None:
        if value:
            self._render_finished = True

    def visualize_goal(self) -> None:
        """Visualize the goal in the renderer (placeholder)."""
        pass

    def inference_mode_post_init_callback(self) -> None:
        """Hook called after creation when the runner is in inference/play mode."""
        pass

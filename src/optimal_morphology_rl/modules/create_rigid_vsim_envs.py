"""Module that creates the rigid-body vlearn simulation and environment group."""

from __future__ import annotations

import math
import random
from typing import Any, List

import numpy as np
import torch
import vlearn as v

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("create_rigid_vsim_envs")
class CreateRigidVsimEnvs(BaseModule):
    """Creates the vlearn Gym, environment definition, and environment group.

    This module is intentionally limited to rigid-body simulation setup.  It
    exposes the resulting handles through the shared
    :attr:`ModuleManager.container` so that ``robot`` and ``object_generator``
    modules can populate the environment definition in their ``finalize`` hooks,
    after which this module finalizes the definition and instantiates the
    environment group in ``post_finalize``.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._validate_config()
        self._apply_seed()
        self._create_gym()

    def _validate_config(self) -> None:
        """Ensure required keys are present."""
        required = ["num_envs", "device"]
        for key in required:
            if key not in self.config:
                raise ValueError(f"create_rigid_vsim_envs config missing '{key}'")

        self.num_envs = int(self.config["num_envs"])
        if self.num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        self.total_num_envs = self.num_envs

        self.device = torch.device(self.config["device"])
        if self.device.type != "cuda":
            raise ValueError("vlearn requires a CUDA device")

    def _apply_seed(self) -> None:
        """Seed RNGs when a seed is provided in config."""
        seed = self.config.get("seed", None)
        if seed is not None:
            seed = int(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

    def _create_gym(self) -> None:
        """Create the vlearn Gym and set global sim parameters."""
        # Simulation frequency / control frequency.
        self.timestep = float(self.config.get("timestep", 1.0 / 120.0))
        self.frame_skip = int(self.config.get("frame_skip", 2))

        # Rendering / window.
        self.rendering = bool(self.config.get("rendering", False))
        self.enable_scene_query = bool(self.config.get("enable_scene_query", True))
        # Default to no window when rendering is not configured by a runner.
        self.with_window = bool(self.config.get("with_window", False))

        # Misc.
        self.spacing = float(self.config.get("spacing", 0.5))
        self.print_hash = bool(self.config.get("print_hash", False))

        # Gravity / up axis.
        gravity = self.config.get("gravity", [0.0, 0.0, -9.81])
        if len(gravity) != 3:
            raise ValueError("gravity must be a list/tuple of 3 floats")
        self.gravity = v.Vec3(*gravity)

        up_axis = self.config.get("up_axis", [0.0, 0.0, 1.0])
        if len(up_axis) != 3:
            raise ValueError("up_axis must be a list/tuple of 3 floats")
        self.up_axis = v.Vec3(*up_axis)
        self.up_axis_rotation = v.shortest_rotation(self.up_axis, v.Vec3(0, 1, 0))

        # Contact limits.
        self.max_contact_pairs_per_env = int(
            self.config.get("max_contact_pairs_per_env", 128)
        )
        max_contact_patches = self.config.get("max_contact_patches_per_env", -1)
        self.max_contact_patches_per_env = (
            self.max_contact_pairs_per_env
            if max_contact_patches == -1
            else int(max_contact_patches)
        )
        self.max_contact_points_per_patch = int(
            self.config.get("max_contact_points_per_patch", 4)
        )

        self.gym = v.create_gym(
            self.rendering,
            self.enable_scene_query,
            treat_warning_as_error=True,
            up_axis=self.up_axis,
            with_window=self.with_window,
            max_contact_pairs=self.max_contact_pairs_per_env * self.total_num_envs,
            max_contact_patches=self.max_contact_patches_per_env * self.total_num_envs,
            max_contact_points=self.max_contact_points_per_patch
            * self.max_contact_patches_per_env
            * self.total_num_envs,
            update_scene_dependent_components_in_step=True,
            cuda_device=self.device.index,
            enable_graph_captures=True,
            seed=self.config.get("seed", None),
            verbose=True,
        )

        self.gym.set_timestep(self.timestep)
        self.gym.set_gravity(self.gravity)

    def finalize(self, container: ModuleContainer) -> None:
        """Create the environment definition and share state in the container."""
        env_def_name = self.config.get("env_def_name", "rigid_env")
        self.env_def_handle = self.gym.create_environment_def(env_def_name)
        self.env_def = self.gym.get_environment_def(self.env_def_handle)

        container.gym = self.gym
        container.env_def_handle = self.env_def_handle
        container.env_def = self.env_def
        container.device = self.device
        container.num_envs = [self.num_envs]
        container.total_num_envs = self.total_num_envs
        container.timestep = self.timestep
        container.frame_skip = self.frame_skip
        container.rendering = self.rendering
        container.with_window = self.with_window
        container.gravity = self.gravity
        container.up_axis = self.up_axis
        container.spacing = self.spacing
        container.max_contact_pairs_per_env = self.max_contact_pairs_per_env
        container.max_contact_patches_per_env = self.max_contact_patches_per_env
        container.max_contact_points_per_patch = self.max_contact_points_per_patch

    def post_finalize(self, container: ModuleContainer) -> None:
        """Finalize the environment definition and create the environment group."""
        if container.get("env_def") is None:
            raise RuntimeError(
                "Environment definition not found in container. "
                "Ensure create_rigid_vsim_envs is listed before dependent modules."
            )

        container.env_def.finalize()

        env_set_offsets = self._compute_env_set_offsets()
        self.env_group = self.gym.create_environment_group(
            self.env_def_handle, [self.num_envs]
        )

        # Arrange environments on a grid inside each environment set.
        self.env_sets = list(self.env_group.get_environment_sets())
        for env_set, offset in zip(self.env_sets, env_set_offsets):
            num_envs_in_set = env_set.get_num_environments()
            grid_size = math.ceil(math.sqrt(num_envs_in_set))
            for i in range(num_envs_in_set):
                x = i % grid_size
                y = i // grid_size
                env_pos = self.up_axis_rotation.rotate(
                    self.spacing * v.Vec3(x, 0, y)
                ) + offset
                env_handle = env_set.get_environment_handle(i)
                environment = env_set.get_environment(env_handle)
                environment.set_transform(v.Transform(v.Quat(0, 0, 0, 1), env_pos))

        self.env_set_handles = list(self.env_group.get_environment_set_handles())
        self.env_set = self.env_sets[-1]
        self.env_set_handle = self.env_set_handles[-1]

        # Share the instantiated group back to the container.
        container.env_group = self.env_group
        container.env_sets = self.env_sets
        container.env_set = self.env_set
        container.env_set_handle = self.env_set_handle
        container.env_set_handles = self.env_set_handles

        self.gym.finalize()
        self.gym.update_scene_dependent_components()

    def _compute_env_set_offsets(self) -> List[v.Vec3]:
        """Return one offset per environment set.

        For now a single environment set centered at the origin is used.  This
        can be extended later to split environments across multiple sets.
        """
        return [v.Vec3(0, 0, 0)]

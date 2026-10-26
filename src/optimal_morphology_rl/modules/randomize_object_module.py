"""Module that randomizes the size, mass and inertia of a rigid object (e.g. the cube)."""

from __future__ import annotations

from typing import Any

import torch
import vlearn as v

from optimal_morphology_rl.modules.base_module import BaseModule, curriculum_range
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


def _scale_range(config: dict[str, Any], key: str) -> tuple[float, float]:
    """Validate a positive ``[min, max]`` scale range (default: no scaling)."""
    value = config.get(key, [1.0, 1.0])
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise RuntimeError(f"RandomizeObjectModule config '{key}' must be a [min, max] list, got {value}.")
    low, high = float(value[0]), float(value[1])
    if not 0.0 < low <= high:
        raise RuntimeError(f"RandomizeObjectModule config '{key}' must satisfy 0 < min <= max, got {value}.")
    return low, high


@register_module("randomize_object")
class RandomizeObjectModule(BaseModule):
    """Randomizes the size, mass and diagonal inertia of a loaded rigid object on reset.

    Per env set, a size scale ``s`` (uniform, all axes), a mass scale ``s_m`` and an inertia scale ``s_i`` are drawn
    from their ranges, and the object gets

        size    = s * defined size
        mass    = s_m * defined_mass
        inertia = s_m * s^2 * s_i * defined_diag_inertia      (all three axes)

    i.e. the inertia follows the mass and the size (same shape), and ``s_i`` adds shape uncertainty on top; the mass
    does not follow the size (``mass_scale_range`` covers it). The size goes through vlearn's rigid body def scale,
    then the mass and inertia are written (after it, as the scale may recompute them). Rigid body defs are shared
    by an env set, so there is one value per env set (per env with ``create_rigid_vsim_envs.num_envs_per_set: 1``);
    a set is resampled when any of its envs resets. All ranges follow the training curriculum from 1 (at 0) to the
    configured range (at 1).

    The object (default: the 'create_objects' reward object) must be listed in 'create_objects' before this
    module.

    Config shape::

        randomize_object:
          object: cube                      # optional
          size_scale_range: [0.9, 1.1]
          mass_scale_range: [0.7, 1.3]
          inertia_scale_range: [0.8, 1.2]
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.object_name = self.config.get("object")
        self.size_range = _scale_range(self.config, "size_scale_range")
        self.mass_range = _scale_range(self.config, "mass_scale_range")
        self.inertia_range = _scale_range(self.config, "inertia_scale_range")

    def post_finalize(self, container: ModuleContainer) -> None:
        objects = container.get("objects")
        if objects is None:
            raise RuntimeError("RandomizeObjectModule requires the 'create_objects' module (listed before it).")
        name = self.object_name or container.reward_object_name
        if name not in objects:
            raise RuntimeError(f"RandomizeObjectModule: object '{name}' is not created (objects: {list(objects)}).")
        obj = objects[name]
        device = container.device
        env_def = container.env_def
        def_handle = env_def.get_rigid_body_def_handle_by_name(obj.name)
        body_def = env_def.get_rigid_body_def(def_handle)
        if body_def.inv_mass <= 0.0:
            raise RuntimeError(f"RandomizeObjectModule: object '{name}' is fixed (infinite mass).")
        self.defined_mass = 1.0 / body_def.inv_mass
        inv = body_def.diag_inv_inertia
        self.defined_inertia = 1.0 / torch.tensor([inv.x, inv.y, inv.z], device=device, dtype=torch.float32)

        num_env_sets = len(container.num_envs)
        # Env set of every env (envs are numbered set by set).
        self.env_set_index = torch.repeat_interleave(
            torch.arange(num_env_sets, device=device), torch.tensor(container.num_envs, device=device)
        )
        self.size_scale = torch.ones(num_env_sets, device=device)
        self.mass_scale = torch.ones(num_env_sets, device=device)
        self.inertia_scale = torch.ones(num_env_sets, device=device)
        self.set_scale_buf = torch.ones((num_env_sets, 3), device=device)
        self.set_inv_mass_buf = torch.full((num_env_sets,), 1.0 / self.defined_mass, device=device)
        self.set_inv_inertia_buf = (1.0 / self.defined_inertia).repeat(num_env_sets, 1)

        group, gym = container.env_group, container.gym
        self.scale_cmd = gym.create_gpu_array(
            [group.create_rigid_body_def_scale_command(v.wrap_gpu_buffer(self.set_scale_buf), def_handle)]
        )
        self.mass_cmd = gym.create_gpu_array(
            [group.create_rigid_body_property_command(v.RigidBodyProperty.INV_MASS, v.wrap_gpu_buffer(self.set_inv_mass_buf), def_handle)]
        )
        self.inertia_cmd = gym.create_gpu_array(
            [
                group.create_rigid_body_property_command(
                    v.RigidBodyProperty.DIAG_INV_INERTIA, v.wrap_gpu_buffer(self.set_inv_inertia_buf), def_handle
                )
            ]
        )
        print(
            f"RandomizeObjectModule: '{name}' size scale {list(self.size_range)}, mass scale {list(self.mass_range)}, "
            f"inertia scale {list(self.inertia_range)} (mass {self.defined_mass * 1000:.1f} g)"
        )

    def reset(self, container: ModuleContainer) -> None:
        """Resample the env sets that contain a resetting env and write size, then mass and inertia."""
        reset_buf = container.reset_buf
        if not reset_buf.any():
            return
        set_mask = torch.zeros_like(self.size_scale, dtype=torch.bool)
        set_mask[self.env_set_index[reset_buf]] = True

        def draw(low_high):
            low, high = curriculum_range(*low_high, self.curriculum_factor, nominal=1.0)
            return low + (high - low) * torch.rand_like(self.size_scale)

        self.size_scale[:] = torch.where(set_mask, draw(self.size_range), self.size_scale)
        self.mass_scale[:] = torch.where(set_mask, draw(self.mass_range), self.mass_scale)
        self.inertia_scale[:] = torch.where(set_mask, draw(self.inertia_range), self.inertia_scale)

        self.set_scale_buf[:] = self.size_scale.unsqueeze(1)
        self.set_inv_mass_buf[:] = 1.0 / (self.mass_scale * self.defined_mass)
        inertia = (self.mass_scale * self.size_scale**2 * self.inertia_scale).unsqueeze(1) * self.defined_inertia
        self.set_inv_inertia_buf[:] = 1.0 / inertia
        gym = container.gym
        gym.scale_rigid_body_defs(self.scale_cmd)
        gym.set_rigid_body_properties(self.mass_cmd)
        gym.set_rigid_body_properties(self.inertia_cmd)

"""Module that randomizes the friction of the robot's rigid material per env set on reset."""

from __future__ import annotations

from typing import Any

import torch
import vlearn as v

from optimal_morphology_rl.modules.base_module import BaseModule, curriculum_range
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("randomize_robot_friction")
class RandomizeRobotFrictionModule(BaseModule):
    """Randomizes the friction between the hand and the objects, per env set, on reset.

    A contact's friction is the average of its two materials' frictions, and the objects' material has friction 0
    ('create_objects'), so the hand's material gets twice the desired contact friction. ``friction_range`` is the
    contact (effective) static friction mu; per env set it is drawn uniformly on reset and written as

        hand static friction  = 2 * mu
        hand dynamic friction = 2 * mu * dynamic_ratio

    Rigid materials are shared by an env set (one value per set; per env with
    ``create_rigid_vsim_envs.num_envs_per_set: 1``); a set is resampled when any of its envs resets. The range
    follows the training curriculum from the nominal contact friction, ``create_robot``'s ``link_friction`` / 2 (at
    0), to ``friction_range`` (at 1).

    Config::

        randomize_robot_friction:
          friction_range: [0.3, 0.7]   # contact static friction mu
          dynamic_ratio: 1.0           # optional (default 1.0, as create_robot): dynamic = ratio * static
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        value = self.config.get("friction_range")
        if not isinstance(value, (list, tuple)) or len(value) != 2 or not 0.0 <= float(value[0]) <= float(value[1]):
            raise RuntimeError(f"RandomizeRobotFrictionModule config 'friction_range' must be [min, max] with 0 <= min <= max, got {value}.")
        self.low, self.high = float(value[0]), float(value[1])
        self.dynamic_ratio = float(self.config.get("dynamic_ratio", 1.0))

    def finalize(self, container: ModuleContainer) -> None:
        if container.get("robot") is None:
            raise RuntimeError(
                "RandomizeRobotFrictionModule requires 'robot' in the shared container. "
                "Ensure the 'create_robot' module is listed before 'randomize_robot_friction'."
            )

    def post_finalize(self, container: ModuleContainer) -> None:
        robot = container.robot
        device = container.device
        self.nominal = robot.link_friction / 2.0
        num_env_sets = len(container.num_envs)
        # Env set of every env (envs are numbered set by set).
        self.env_set_index = torch.repeat_interleave(
            torch.arange(num_env_sets, device=device), torch.tensor(container.num_envs, device=device)
        )
        self.friction = torch.full((num_env_sets,), self.nominal, device=device)  # contact friction per env set
        self.set_static_buf = torch.full((num_env_sets,), robot.link_friction, device=device)
        self.set_dynamic_buf = torch.full((num_env_sets,), robot.link_friction, device=device)
        self.set_mask = torch.zeros(num_env_sets, dtype=torch.bool, device=device)
        mask = v.wrap_gpu_buffer(self.set_mask)
        commands = [
            container.env_group.create_rigid_material_property_command(
                v.RigidMaterialProperty.STATIC_FRICTION, v.wrap_gpu_buffer(self.set_static_buf), robot.rigid_mat_handle, mask
            ),
            container.env_group.create_rigid_material_property_command(
                v.RigidMaterialProperty.DYNAMIC_FRICTION, v.wrap_gpu_buffer(self.set_dynamic_buf), robot.rigid_mat_handle, mask
            ),
        ]
        self.command_array = container.gym.create_gpu_array(commands)
        print(
            f"RandomizeRobotFrictionModule: contact friction {[self.low, self.high]} (nominal {self.nominal:g}), "
            f"dynamic ratio {self.dynamic_ratio:g}, {num_env_sets} env sets"
        )

    def reset(self, container: ModuleContainer) -> None:
        """Resample the env sets that contain a resetting env and write their hand friction."""
        reset_buf = container.reset_buf
        if not reset_buf.any():
            return
        self.set_mask[:] = False
        self.set_mask[self.env_set_index[reset_buf]] = True
        low, high = curriculum_range(self.low, self.high, self.curriculum_factor, nominal=self.nominal)
        samples = low + (high - low) * torch.rand_like(self.friction)
        self.friction[:] = torch.where(self.set_mask, samples, self.friction)
        self.set_static_buf[:] = 2.0 * self.friction
        self.set_dynamic_buf[:] = 2.0 * self.friction * self.dynamic_ratio
        container.gym.set_rigid_material_properties(self.command_array)

"""Module that randomizes and applies the rigid material friction coefficients."""

from __future__ import annotations

from typing import Any

import torch
import vlearn as v

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("randomize_friction")
class RandomizeFrictionModule(BaseModule):
    """Randomizes and applies the rigid material friction coefficients.

    On every reset the friction is randomized (multi-env runs without a
    configured coefficient) or taken from the ``friction_coefficient`` config
    (default ``None`` -> ``0.1``), then applied via
    ``gym.set_rigid_material_properties``.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.fric_coeff = self.config.get("friction_coefficient", None)

    def finalize(self, container: ModuleContainer) -> None:
        """Validate that the robot is available."""
        if container.get("robot") is None:
            raise RuntimeError(
                "RandomizeFrictionModule requires 'robot' in the shared container. "
                "Ensure the 'create_robot' module is listed before 'randomize_friction'."
            )

    def post_finalize(self, container: ModuleContainer) -> None:
        """Allocate the friction buffers and create the GPU command."""
        robot = container.robot
        device = container.device

        # Rigid material property buffers are scalar per material.
        container.set_static_friction_buf = torch.zeros(1, dtype=torch.float32, device=device)
        container.set_dynamic_friction_buf = torch.zeros(1, dtype=torch.float32, device=device)

        set_static_friction_cmd = container.env_group.create_rigid_material_property_command(
            v.RigidMaterialProperty.STATIC_FRICTION,
            v.wrap_gpu_buffer(container.set_static_friction_buf),
            robot.rigid_mat_handle,
            v.wrap_gpu_buffer(container.reset_buf),
        )
        set_dynamic_friction_cmd = container.env_group.create_rigid_material_property_command(
            v.RigidMaterialProperty.DYNAMIC_FRICTION,
            v.wrap_gpu_buffer(container.set_dynamic_friction_buf),
            robot.rigid_mat_handle,
            v.wrap_gpu_buffer(container.reset_buf),
        )
        container.gpu_set_friction_cmd = container.gym.create_gpu_array([set_static_friction_cmd, set_dynamic_friction_cmd])

    def reset(self, container: ModuleContainer) -> None:
        """Randomize and apply the friction coefficients for the resetting envs."""
        reset_buf = container.reset_buf
        device = container.device
        gym = container.gym

        total_num_envs = reset_buf.shape[0]
        if total_num_envs != 1 and self.fric_coeff is None:
            static_friction = torch.rand(1, device=device).item() * 0.9 + 0.1
        else:
            static_friction = 0.1 if self.fric_coeff is None else self.fric_coeff
        dynamic_friction = static_friction * 0.75

        container.set_static_friction_buf[0] = static_friction * 2.0
        container.set_dynamic_friction_buf[0] = dynamic_friction * 2.0
        gym.set_rigid_material_properties(container.gpu_set_friction_cmd)

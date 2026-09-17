"""Tendon force application via gym.set_spatial_tendon_forces."""

from __future__ import annotations

import torch
import vlearn as v

from optimal_morphology_rl.modules.apply_control.helpers import require_robot
from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("apply_tendon_forces")
class ApplyTendonForcesModule(BaseModule):
    """Owns the tendon control buffer/GPU command and applies the tendon forces.

    The control values are composed by ``robot_control_tendons``; this module
    applies them via ``gym.set_spatial_tendon_forces``.
    """

    def finalize(self, container: ModuleContainer) -> None:
        """Validate that the robot is available."""
        require_robot(container, "apply_tendon_forces")

    def post_finalize(self, container: ModuleContainer) -> None:
        """Allocate the tendon control buffer and create the GPU command."""
        robot = container.robot
        device = container.device

        container.set_tendon_controls_buf = torch.zeros(
            (container.total_num_envs, robot.num_tendons), device=device, dtype=torch.float32
        )
        set_tendon_cmd = container.env_group.create_spatial_tendon_control_command(
            v.wrap_gpu_buffer(container.set_tendon_controls_buf), robot.arti_handle
        )
        container.gpu_set_tendon_control_command_array = container.gym.create_gpu_array([set_tendon_cmd])

    def step(self, container: ModuleContainer) -> None:
        """Apply the composed tendon control commands."""
        container.gym.set_spatial_tendon_forces(container.gpu_set_tendon_control_command_array)

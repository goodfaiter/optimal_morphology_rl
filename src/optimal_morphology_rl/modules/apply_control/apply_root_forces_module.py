"""Root force/torque application via gym.set_link_external_forces."""

from __future__ import annotations

import torch
import vlearn as v

from optimal_morphology_rl.modules.apply_control.helpers import require_robot
from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("apply_root_forces")
class ApplyRootForcesModule(BaseModule):
    """Owns the root force/torque buffer/GPU command and applies the root forces.

    The gravity compensation forces are computed by ``gravity_compensation``;
    this module applies them via ``gym.set_link_external_forces``.
    """

    def finalize(self, container: ModuleContainer) -> None:
        """Validate that the robot is available."""
        require_robot(container, "apply_root_forces")

    def post_finalize(self, container: ModuleContainer) -> None:
        """Allocate the force/torque buffer and create the GPU command."""
        robot = container.robot

        container.set_force_torque_buf = torch.zeros(
            (container.total_num_envs, robot.num_links, 6), device=container.device, dtype=torch.float32
        )
        set_force_torque_cmd = container.env_group.create_link_external_force_command(
            v.wrap_gpu_buffer(container.set_force_torque_buf),
            robot.arti_handle,
            [0, robot.num_links],
            force_type=v.ForceType.FORCE_TORQUE,
        )
        container.set_force_torque_cmd_arr = container.gym.create_gpu_array([set_force_torque_cmd])

    def step(self, container: ModuleContainer) -> None:
        """Apply the composed root force/torque commands."""
        container.gym.set_link_external_forces(container.set_force_torque_cmd_arr)

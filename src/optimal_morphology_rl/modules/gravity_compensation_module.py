"""Module that computes gravity compensation forces on the base link."""

from __future__ import annotations

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("gravity_compensation")
class GravityCompensationModule(BaseModule):
    """Computes gravity compensation forces on the base link.

    The computed forces are written to ``container.set_force_torque_buf`` and
    applied by ``apply_root_forces`` via ``gym.set_link_external_forces``.
    """

    def finalize(self, container: ModuleContainer) -> None:
        """Validate that the robot is available."""
        if container.get("robot") is None:
            raise RuntimeError(
                "GravityCompensationModule requires 'robot' in the shared container. "
                "Ensure the 'create_robot' module is listed before 'gravity_compensation'."
            )

    def step(self, container: ModuleContainer) -> None:
        """Compute the gravity compensation force on the base link."""
        robot = container.robot
        container.set_force_torque_buf[:, :, 2] = 9.81 * robot.link_masses

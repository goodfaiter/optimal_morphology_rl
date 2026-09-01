"""Module that refreshes robot state buffers after the physics step."""

from __future__ import annotations

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("update_robot")
class UpdateRobotModule(BaseModule):
    """Refreshes robot state buffers for the robot created by ``create_robot``."""

    def step(self, container: ModuleContainer) -> None:
        """Refresh robot state buffers."""
        robot = container.get("robot")
        if robot is None:
            return
        robot.refresh_buffers(container.gym)

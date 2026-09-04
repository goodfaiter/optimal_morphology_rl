"""Module that refreshes the kinematic sensor after the physics step."""

from __future__ import annotations

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("update_kinematic_sensor")
class UpdateKinematicSensorModule(BaseModule):
    """Refreshes the reward-object kinematic sensor created during finalize."""

    def step(self, container: ModuleContainer) -> None:
        """Refresh kinematic sensor buffers and compute robot-frame state."""
        kinematic_sensor = container.get("kinematic_sensor")
        if kinematic_sensor is None:
            return
        kinematic_sensor.update(container.gym)

        robot_state = container.get("robot_state")
        if robot_state is not None:
            kinematic_sensor.update_robot_frame(robot_state)

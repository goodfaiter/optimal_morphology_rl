"""Module that refreshes the kinematic sensors after the physics step."""

from __future__ import annotations

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("update_kinematic_sensor")
class UpdateKinematicSensorModule(BaseModule):
    """Refreshes every created kinematic sensor after the physics step."""

    def step(self, container: ModuleContainer) -> None:
        """Refresh kinematic sensor buffers and compute robot-frame state."""
        sensors = container.get("kinematic_sensors") or {}
        robot_state = container.get("robot_state")
        for sensor in sensors.values():
            sensor.update(container.gym)
            if robot_state is not None:
                sensor.update_robot_frame(robot_state)

"""Module that refreshes the kinematic sensors after the physics step."""

from __future__ import annotations

import torch

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("update_kinematic_sensor")
class UpdateKinematicSensorModule(BaseModule):
    """Refreshes every created kinematic sensor after the physics step."""

    def step(self, container: ModuleContainer) -> None:
        """Refresh kinematic sensor buffers and compute robot-frame state."""
        self._refresh(container, None)

    def reset(self, container: ModuleContainer) -> None:
        """Read the reset envs' sensors: forward kinematics updates their transforms; their velocities are the reset
        state's (see KinematicSensor.update)."""
        container.gym.compute_kinematics()
        self._refresh(container, container.get("reset_buf"))

    @staticmethod
    def _refresh(container: ModuleContainer, reset_rows: torch.Tensor | None) -> None:
        sensors = container.get("kinematic_sensors") or {}
        robot_state = container.get("robot_state")
        for sensor in sensors.values():
            sensor.update(container.gym, reset_rows)
            if robot_state is not None:
                sensor.update_robot_frame(robot_state)

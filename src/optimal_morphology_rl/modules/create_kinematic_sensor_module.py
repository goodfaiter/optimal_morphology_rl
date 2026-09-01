"""Module that creates a kinematic sensor on the reward object."""

from __future__ import annotations

from typing import Any

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.kinematic_sensor import KinematicSensor
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("create_kinematic_sensor")
class CreateKinematicSensorModule(BaseModule):
    """Creates a kinematic sensor on the reward object.

    Expects ``container.reward_object`` and ``container.env_def``.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.sensor = KinematicSensor()

    def finalize(self, container: ModuleContainer) -> None:
        """Verify dependencies."""
        if container.get("reward_object") is None:
            raise RuntimeError(
                "CreateKinematicSensorModule requires 'reward_object' in the shared container."
            )
        if container.get("env_def") is None:
            raise RuntimeError(
                "CreateKinematicSensorModule requires 'env_def' in the shared container."
            )

    def post_finalize(self, container: ModuleContainer) -> None:
        """Allocate buffers and create GPU commands."""
        # Buffer allocation must happen after the environment definition has been
        # finalized (it resolves the per-instance kinematic sensor handle).
        self.sensor.allocate_buffers(
            container.env_def,
            container.reward_object.handle,
            container.total_num_envs,
            container.device,
        )
        container.kinematic_sensor = self.sensor
        self.sensor.create_gpu_commands(container.env_group, container.gym)

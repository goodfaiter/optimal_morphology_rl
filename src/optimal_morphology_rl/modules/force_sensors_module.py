"""Module that discovers and reads fingertip force sensors on the robot."""

from __future__ import annotations

from typing import Any, Sequence

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.force_sensors import ForceSensors
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("force_sensors")
class ForceSensorsModule(BaseModule):
    """Wraps ForceSensors and exposes the buffer on the shared container."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.sensors = ForceSensors()
        self.link_names: Sequence[str] = self.config.get("link_names", ["distal"])

    def finalize(self, env: Any) -> None:
        container = env.module_manager.container
        if container.get("robot") is None or container.get("env_def") is None:
            raise RuntimeError(
                "ForceSensorsModule requires 'robot' and 'env_def' in the shared container."
            )

    def post_finalize(self, env: Any) -> None:
        container = env.module_manager.container
        env_def = container.env_def
        robot = container.robot
        # The articulation instance is only available after env_def.finalize().
        articulation = env_def.get_articulation(robot.arti_handle)

        self.sensors.allocate_buffers(
            robot.art_def,
            articulation,
            env.total_num_envs,
            env.device,
            link_names=self.link_names,
        )
        container.force_sensors = self.sensors
        self.sensors.create_gpu_commands(container.env_group, container.gym)

    def post_physics_step(self, env: Any) -> None:
        self.sensors.update(env.module_manager.container.gym)

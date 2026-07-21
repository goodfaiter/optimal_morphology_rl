"""Module that loads the robot hand articulation into the environment definition."""

from __future__ import annotations

from typing import Any

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_manager import register_module
from optimal_morphology_rl.modules.robot import Robot


@register_module("robot")
class RobotModule(BaseModule):
    """Loads and exposes the robot hand.

    Expects the shared container to already contain ``env_def`` and ``device``
    (populated by ``create_rigid_vsim_envs``).
    """

    def finalize(self, env: Any) -> None:
        container = env.module_manager.container

        if container.get("env_def") is None:
            raise RuntimeError(
                "RobotModule requires 'env_def' in the shared container. "
                "Ensure create_rigid_vsim_envs is listed before robot."
            )
        if container.get("device") is None:
            raise RuntimeError(
                "RobotModule requires 'device' in the shared container."
            )

        vsim_path = self.config.get("vsim_path")
        if vsim_path is None:
            raise ValueError("robot config missing 'vsim_path'")

        fixed_hand = bool(self.config.get("fixed_hand", False))
        use_tendon = bool(self.config.get("use_tendon", True))

        self.robot = Robot(fixed_hand=fixed_hand, use_tendon=use_tendon)
        self.robot.create_envs(container.env_def, vsim_path, container.device)

        container.robot = self.robot
        container.robot_vsim_path = vsim_path

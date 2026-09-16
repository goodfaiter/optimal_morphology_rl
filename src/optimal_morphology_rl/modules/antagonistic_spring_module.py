"""Module that computes per-motor antagonistic spring forces."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


def _build_motor_scale_tensor(cfg_value: Any, expected_len: int, device: torch.device, name: str) -> torch.Tensor:
    """Convert a scalar or per-motor list config value to a float tensor."""
    if isinstance(cfg_value, (list, tuple)):
        if len(cfg_value) != expected_len:
            raise RuntimeError(f"AntagonisticSpring config '{name}' length ({len(cfg_value)}) must match {expected_len}.")
        return torch.tensor(cfg_value, device=device, dtype=torch.float32)
    return torch.full((expected_len,), float(cfg_value), device=device, dtype=torch.float32)


@register_module("antagonistic_spring")
class AntagonisticSpring(BaseModule):
    """Computes per-motor antagonistic spring forces.

    The module applies a linear spring-like resistance around the joint zero
    position for every motor:

    ``force = -spring_constants * joint_pos[motor_to_joint_dof_index]``

    The computed force is written to
    ``container.antagonistic_spring_force_buf`` so ``robot_control`` can add it
    to the motor commands before the forces are applied via
    ``set_motor_forces``.
    """

    def finalize(self, container: ModuleContainer) -> None:
        """Validate cross-module dependencies."""
        if container.get("robot") is None:
            raise RuntimeError(
                "AntagonisticSpring requires 'robot' in the shared container. "
                "Ensure the 'create_robot' module is listed before 'antagonistic_spring'."
            )
        if container.get("env") is None:
            raise RuntimeError("AntagonisticSpring requires 'env' in the shared container.")

    def post_finalize(self, container: ModuleContainer) -> None:
        """Build the spring constants and allocate the force buffer."""
        robot = container.robot
        device = container.device

        spring_constants = self.config.get("spring_constants")
        if spring_constants is None:
            raise RuntimeError("AntagonisticSpring config missing 'spring_constants': a scalar or per-motor list of spring constants.")
        self.spring_constants = _build_motor_scale_tensor(spring_constants, robot.num_motors, device, "spring_constants")

        container.antagonistic_spring_force_buf = torch.zeros(
            (container.total_num_envs, robot.num_motors), device=device, dtype=torch.float32
        )

    def step(self, container: ModuleContainer) -> None:
        """Compute the antagonistic spring force for each motor."""
        robot = container.robot

        container.antagonistic_spring_force_buf[:] = -self.spring_constants * robot.get_joint_pos_buf[:, robot.motor_to_joint_dof_index]

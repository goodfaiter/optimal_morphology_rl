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


def _validate_spring_range(cfg_value: Any) -> tuple[float, float]:
    """Validate a domain-randomization range and return it as floats."""
    if not isinstance(cfg_value, (list, tuple)) or len(cfg_value) != 2:
        raise RuntimeError(f"AntagonisticSpring config 'spring_constant_range' must be a [min, max] list, got {cfg_value}.")
    spring_min, spring_max = float(cfg_value[0]), float(cfg_value[1])
    if spring_min > spring_max:
        raise RuntimeError(f"AntagonisticSpring config 'spring_constant_range' min ({spring_min}) must be <= max ({spring_max}).")
    return spring_min, spring_max


@register_module("antagonistic_spring")
class AntagonisticSpring(BaseModule):
    """Computes per-motor antagonistic spring forces.

    The module applies a linear spring-like resistance around the joint zero
    position for every motor on a revolute joint (motors on other joints,
    e.g. the prismatic tendon actuator sliders, get no spring force):

    ``force = -spring_constants * joint_pos[motor_to_joint_dof_index]``

    Domain randomization: when a ``spring_constant_range`` config (``[min,
    max]``) is given, per-env spring constants are sampled uniformly from that
    range at every reset (resampling only the resetting envs); the
    ``spring_constants`` default is then optional. Without the range key,
    every env uses the ``spring_constants`` default (required then).

    The computed force is written to
    ``container.antagonistic_spring_force_buf`` so ``robot_control`` can add it
    to the motor commands before the forces are applied via
    ``set_motor_forces``.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.spring_constants: torch.Tensor | None = None
        self.spring_min: float | None = None
        self.spring_max: float | None = None

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

        spring_range = self.config.get("spring_constant_range")
        if spring_range is not None:
            self.spring_min, self.spring_max = _validate_spring_range(spring_range)

        spring_constants = self.config.get("spring_constants")
        if spring_constants is None:
            if spring_range is None:
                raise RuntimeError(
                    "AntagonisticSpring config missing 'spring_constants': a scalar or per-motor list of spring constants, "
                    "or provide 'spring_constant_range'."
                )
        else:
            self.spring_constants = _build_motor_scale_tensor(spring_constants, robot.num_motors, device, "spring_constants")

        if spring_range is not None:
            self.spring_constants_buf = self._sample_constants(container.total_num_envs, robot.num_motors, device)
        else:
            self.spring_constants_buf = self.spring_constants.repeat(container.total_num_envs, 1)

        self.revolute_motor_mask = robot.revolute_dof_mask[robot.motor_to_joint_dof_index]

        container.antagonistic_spring_force_buf = torch.zeros(
            (container.total_num_envs, robot.num_motors), device=device, dtype=torch.float32
        )
        parts = []
        if spring_range is not None:
            parts.append(f"spring_constant_range = [{self.spring_min}, {self.spring_max}]")
        if self.spring_constants is not None:
            parts.append(f"spring_constants = {self.spring_constants.cpu().numpy()}")
        names = [robot.art_def.get_motor_def(i).name for i in range(robot.num_motors) if self.revolute_motor_mask[i]]
        parts.append(f"revolute motors = {names}")
        print(f"AntagonisticSpring: {', '.join(parts)}")

    def _sample_constants(self, num_rows: int, num_motors: int, device: torch.device) -> torch.Tensor:
        """Sample spring constants uniformly from the configured range."""
        return torch.rand((num_rows, num_motors), device=device) * (self.spring_max - self.spring_min) + self.spring_min

    def reset(self, container: ModuleContainer) -> None:
        """Resample the per-env spring constants of the resetting envs."""
        if self.spring_min is None:
            return
        num_reset = container.reset_buf.sum().item()
        if num_reset == 0:
            return
        self.spring_constants_buf[container.reset_buf] = self._sample_constants(
            num_reset, self.spring_constants_buf.shape[1], container.device
        )

    def step(self, container: ModuleContainer) -> None:
        """Compute the antagonistic spring force for each motor."""
        robot = container.robot

        force = -self.spring_constants_buf * robot.get_joint_pos_buf[:, robot.motor_to_joint_dof_index]
        container.antagonistic_spring_force_buf[:] = force * self.revolute_motor_mask

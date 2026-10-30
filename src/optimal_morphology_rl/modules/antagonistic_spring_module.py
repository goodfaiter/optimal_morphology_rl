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
    position for every motor on a revolute joint (motors on other joints,
    e.g. the prismatic tendon actuator sliders, get no spring force):

    ``force = -(spring_constants * joint_pos[motor_to_joint_dof_index] + spring_preloads)``

    ``spring_constants`` (Nm/rad) is a scalar, a list with one value per motor, or a list
    with one value per revolute motor in motor order (e.g. ``[0.1, 1.0, 1.0]``
    for MCP, PIP, DIP; the other motors get 0). ``spring_preloads`` (Nm, optional, default 0, same
    forms) is the torque of a pre-tensioned elastic: it holds the joint against its extension stop
    (the lower limit, q = 0) until a larger torque flexes it, e.g. the thumb against gravity. Both are
    non-negative: the elastic only ever resists flexion, it never helps the motor.

    Both also take a dict per joint, keyed by the revolute motor (= joint) name, with an optional ``default`` for the
    joints not listed (else 0), e.g. ``{default: 0.005, Thumb_MCP: 0.01}``; unknown names raise.

    The constants are kept per env in ``spring_constants_buf`` ``(num_envs,
    num_motors)``, all at ``spring_constants`` unless 'randomize_antagonistic_springs'
    resamples them on reset.

    The computed force is written to
    ``container.antagonistic_spring_force_buf`` so ``robot_control`` can add it
    to the motor commands before the forces are applied via
    ``set_motor_forces``.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.spring_constants: torch.Tensor | None = None

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
            raise RuntimeError(
                "AntagonisticSpring config missing 'spring_constants': a scalar, a per-motor list, a per-revolute-motor "
                "list or a per-joint dict of spring constants."
            )
        self.revolute_motor_mask = robot.revolute_dof_mask[robot.motor_to_joint_dof_index]
        num_revolute = int(self.revolute_motor_mask.sum())

        motor_names = [robot.art_def.get_motor_def(i).name for i in range(robot.num_motors)]

        def per_motor(value, name):
            if isinstance(value, dict):
                revolute = {n for n, rev in zip(motor_names, self.revolute_motor_mask.tolist()) if rev}
                unknown = set(value) - revolute - {"default"}
                if unknown:
                    raise RuntimeError(f"AntagonisticSpring '{name}': {sorted(unknown)} are not revolute motors {sorted(revolute)}.")
                default = float(value.get("default", 0.0))
                values = [float(value.get(n, default)) if n in revolute else 0.0 for n in motor_names]
                return torch.tensor(values, device=device, dtype=torch.float32)
            if isinstance(value, (list, tuple)) and len(value) == num_revolute != robot.num_motors:
                # One value per revolute motor: scatter onto the revolute motor columns.
                out = torch.zeros(robot.num_motors, device=device, dtype=torch.float32)
                out[self.revolute_motor_mask] = torch.tensor(value, device=device, dtype=torch.float32)
                return out
            return _build_motor_scale_tensor(value, robot.num_motors, device, name)

        self.spring_constants = per_motor(spring_constants, "spring_constants")
        self.spring_constants_buf = self.spring_constants.repeat(container.total_num_envs, 1)
        self.spring_preloads = per_motor(self.config.get("spring_preloads", 0.0), "spring_preloads")
        if torch.any(self.spring_constants < 0.0) or torch.any(self.spring_preloads < 0.0):
            raise RuntimeError("AntagonisticSpring 'spring_constants' and 'spring_preloads' must be non-negative (the elastic only resists flexion).")

        container.antagonistic_spring_force_buf = torch.zeros(
            (container.total_num_envs, robot.num_motors), device=device, dtype=torch.float32
        )
        names = [n for n, rev in zip(motor_names, self.revolute_motor_mask.tolist()) if rev]
        print(f"AntagonisticSpring: spring_constants = {self.spring_constants.cpu().numpy()}, "
              f"spring_preloads = {self.spring_preloads.cpu().numpy()}, revolute motors = {names}")

    def step(self, container: ModuleContainer) -> None:
        """Compute the antagonistic spring force for each motor."""
        robot = container.robot

        force = -(self.spring_constants_buf * robot.get_joint_pos_buf[:, robot.motor_to_joint_dof_index] + self.spring_preloads)
        container.antagonistic_spring_force_buf[:] = force * self.revolute_motor_mask

"""Active motor control computation for the apply_motor_forces module."""

from __future__ import annotations

import torch

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module
from optimal_morphology_rl.modules.robot_control.helpers import (
    build_action_space,
    validate_action_buffers,
    validate_robot_dependencies,
)


@register_module("robot_control_motors")
class RobotControlMotorsModule(BaseModule):
    """Computes the policy motor force buffer for the apply_motor_forces module.

    Writes the scaled policy actions, clamped to the active motors, into
    ``container.motor_policy_force_buf`` (passive motors receive no policy
    force). apply_motor_forces composes this buffer with the antagonistic
    spring forces when present and applies the total via
    ``gym.set_motor_forces``.
    """

    def finalize(self, container: ModuleContainer) -> None:
        """Build the active motor mask and set the environment action space."""
        validate_robot_dependencies(container, "robot_control_motors")
        if container.robot.use_tendon:
            raise RuntimeError("robot_control_motors is only supported for motor-driven hands.")
        build_action_space(container)

    def post_finalize(self, container: ModuleContainer) -> None:
        """Validate the action buffers and allocate the policy force buffer."""
        validate_action_buffers(container, "robot_control_motors")

        container.motor_policy_force_buf = torch.zeros(
            (container.total_num_envs, container.robot.num_motors),
            device=container.device,
            dtype=torch.float32,
        )

    def step(self, container: ModuleContainer) -> None:
        """Write the active motor policy forces into the policy force buffer."""
        container.motor_policy_force_buf[:, container.active_dof_mask] = torch.clamp(
            container.scaled_act_buf[:, container.active_dof_slice], 0.0, None
        )

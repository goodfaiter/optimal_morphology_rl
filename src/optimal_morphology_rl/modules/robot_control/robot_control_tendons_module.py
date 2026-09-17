"""Tendon control composition for the apply_tendon_forces module."""

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


@register_module("robot_control_tendons")
class RobotControlTendonsModule(BaseModule):
    """Composes the tendon control commands for the apply_tendon_forces module.

    Clamps the scaled policy actions onto the spatial tendons, overrides the
    model-driven tendons with the tendon_est forces when present, and adds the
    rigid_tendons stretch forces when present. The antagonistic spring motor
    forces are added by apply_motor_forces.
    """

    def finalize(self, container: ModuleContainer) -> None:
        """Build the active motor mask and set the environment action space."""
        validate_robot_dependencies(container, "robot_control_tendons")
        if not container.robot.use_tendon:
            raise RuntimeError("robot_control_tendons is only supported for tendon-driven hands.")
        build_action_space(container)

    def post_finalize(self, container: ModuleContainer) -> None:
        """Validate the action buffers are allocated."""
        validate_action_buffers(container, "robot_control_tendons")

    def step(self, container: ModuleContainer) -> None:
        """Compose the tendon control commands."""
        tendons = container.set_tendon_controls_buf
        # Zero the full buffer, then scatter the policy clamp onto the
        # policy-controlled tendon columns only (fixed tendons stay at zero
        # and are driven by modules like 'rigid_tendons').
        tendons[:] = 0.0
        tendons[:, container.active_dof_indices] = torch.clamp(
            container.scaled_act_buf[:, container.active_dof_slice], 0.0, None
        )

        if container.get("tendon_force_buf") is not None and container.get("tendon_model_indices") is not None:
            # Model-driven tendons (see the 'tendon_est' module) are overridden with the model forces.
            tendon_indices = container.tendon_model_indices
            print(tendon_indices)
            container.set_tendon_controls_buf[:, tendon_indices] = container.tendon_force_buf[:, tendon_indices]

        if container.get("rigid_tendon_force_buf") is not None and container.get("rigid_tendon_indices") is not None:
            # Rigid tendons (see the 'rigid_tendons' module) add a stretch-restoring force.
            rigid_indices = container.rigid_tendon_indices
            container.set_tendon_controls_buf[:, rigid_indices] += container.rigid_tendon_force_buf[:, rigid_indices]

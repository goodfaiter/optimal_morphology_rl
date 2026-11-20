"""Reads the joint servo state of 'create_motor' from the simulator."""

from __future__ import annotations

import torch

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("update_motor")
class UpdateMotorModule(BaseModule):
    """Fills the state of the 'create_motor' servos (``container.motor``) every physics step: theta = q and omega = the
    finite-difference qdot (the commanded reset state on the step after a reset). List it before 'apply_motor_servo' in
    ``pre_gym_step_modules`` (with 'update_robot' in ``post_gym_step_modules``)."""

    physics_rate = True  # the finite-difference velocity is over one physics step

    def finalize(self, container: ModuleContainer) -> None:
        if container.get("motor") is None:
            raise RuntimeError("update_motor requires 'create_motor' (container.motor) in init_modules.")

    def reset(self, container: ModuleContainer) -> None:
        container.motor.just_reset |= container.reset_buf

    def step(self, container: ModuleContainer) -> None:
        motor = container.motor
        pos = container.robot.get_joint_pos_buf[:, motor.dof_indices]
        vel = motor.joint_velocity(container)
        if bool(torch.any(motor.just_reset)):
            pos = torch.where(motor.just_reset.unsqueeze(-1), container.reset_joint_pos_buf[:, motor.dof_indices], pos)
            motor.just_reset[:] = False
        motor.last_pos[:] = pos
        # New tensors every step, so references kept by readers (e.g. last_signals) stay valid.
        motor.theta = pos
        motor.omega = vel

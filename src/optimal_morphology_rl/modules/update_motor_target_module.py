"""Fixes the desired motor angle once per control step (the motor's hold: target)."""

from __future__ import annotations

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("update_motor_target")
class UpdateMotorTargetModule(BaseModule):
    """With this module the 'create_motor' servos hold a target (``motor.hold = "target"``): once per control step the
    desired motor angle is fixed at the current joint angle plus the policy's action, and over that control step's
    physics substeps 'apply_motor_servo' moves the joint to it. List it in ``pre_physics_step_modules`` after
    'process_variable_actions'."""

    def finalize(self, container: ModuleContainer) -> None:
        if container.get("motor") is None:
            raise RuntimeError("update_motor_target requires 'create_motor' (container.motor) in init_modules.")
        container.motor.hold = "target"

    def step(self, container: ModuleContainer) -> None:
        motor = container.motor
        angle = container.robot.get_joint_pos_buf[:, motor.dof_indices]
        motor.target = angle + motor.action_delta_angle(container)

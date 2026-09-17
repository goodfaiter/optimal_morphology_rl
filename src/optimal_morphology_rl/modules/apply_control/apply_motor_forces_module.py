"""Motor force application via gym.set_motor_forces."""

from __future__ import annotations

import torch
import vlearn as v

from optimal_morphology_rl.modules.apply_control.helpers import require_robot
from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("apply_motor_forces")
class ApplyMotorForcesModule(BaseModule):
    """Owns the motor command buffer/GPU command and applies the motor forces.

    Composes the total motor force as ``zero + policy forces (when
    robot_control_motors is wired) + antagonistic spring forces (when
    present)`` and applies it via ``gym.set_motor_forces``.
    """

    def finalize(self, container: ModuleContainer) -> None:
        """Validate that the robot is available."""
        require_robot(container, "apply_motor_forces")

    def post_finalize(self, container: ModuleContainer) -> None:
        """Allocate the motor command buffer and create the GPU command."""
        robot = container.robot
        device = container.device

        container.set_motor_cmd_buf = torch.zeros(
            (container.total_num_envs, robot.num_motors), device=device, dtype=torch.float32
        )
        set_motor_cmd = container.env_group.create_motor_control_command(
            v.wrap_gpu_buffer(container.set_motor_cmd_buf),
            robot.arti_handle,
            index_range=[0, robot.num_motors],
        )
        container.gpu_set_motor_control_command_array = container.gym.create_gpu_array([set_motor_cmd])

    def step(self, container: ModuleContainer) -> None:
        """Compose and apply the total motor forces."""
        container.set_motor_cmd_buf[:] = 0.0

        if container.get("motor_policy_force_buf") is not None:
            # Policy motor forces (see the 'robot_control_motors' module).
            container.set_motor_cmd_buf[:] += container.motor_policy_force_buf

        if container.get("antagonistic_spring_force_buf") is not None:
            # Antagonistic spring forces (see the 'antagonistic_spring' module) are applied to all motors.
            container.set_motor_cmd_buf[:] += container.antagonistic_spring_force_buf

        container.gym.set_motor_forces(container.gpu_set_motor_control_command_array)

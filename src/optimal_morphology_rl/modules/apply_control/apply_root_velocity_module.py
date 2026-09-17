"""Root velocity application via gym.set_articulation_kinematic_states."""

from __future__ import annotations

import torch
import vlearn as v

from optimal_morphology_rl.modules.apply_control.helpers import require_robot
from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("apply_root_velocity")
class ApplyRootVelocityModule(BaseModule):
    """Owns the root/joint kinematic buffers/GPU command and sets the kinematic states.

    The root transform/velocities are computed by ``robot_control_floating_hand``;
    this module applies them via ``gym.set_articulation_kinematic_states``.
    """

    def finalize(self, container: ModuleContainer) -> None:
        """Validate that the robot is available."""
        require_robot(container, "apply_root_velocity")

    def post_finalize(self, container: ModuleContainer) -> None:
        """Allocate the root/joint buffers and create the GPU command."""
        if container.get("inverse_reset_buf") is None:
            raise RuntimeError("apply_root_velocity requires 'inverse_reset_buf' in the shared container. Ensure 'termination' is loaded.")

        robot = container.robot
        device = container.device
        total_num_envs = container.total_num_envs

        container.set_joint_pos_buf = torch.zeros((total_num_envs, 0), device=device, dtype=torch.float32)
        container.set_joint_vel_buf = torch.zeros((total_num_envs, 0), device=device, dtype=torch.float32)
        container.set_root_transform_buf = torch.zeros((total_num_envs, 7), device=device, dtype=torch.float32)
        container.set_root_vel_buf = torch.zeros((total_num_envs, 6), device=device, dtype=torch.float32)

        set_kin_cmd = container.env_group.create_articulation_kinematic_state_command(
            v.wrap_gpu_buffer(container.set_joint_pos_buf),
            v.wrap_gpu_buffer(container.set_joint_vel_buf),
            v.wrap_gpu_buffer(container.set_root_transform_buf),
            v.wrap_gpu_buffer(container.set_root_vel_buf),
            robot.arti_handle,
            (0, 0),
            (0, 1),
            masks_buffer=v.wrap_gpu_buffer(container.inverse_reset_buf),
        )
        container.gpu_set_kinematic_state_command_array = container.gym.create_gpu_array([set_kin_cmd])

    def step(self, container: ModuleContainer) -> None:
        """Apply the computed root velocities as kinematic states."""
        container.gym.set_articulation_kinematic_states(container.gpu_set_kinematic_state_command_array)

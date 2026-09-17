"""Floating-hand control computation for the apply_root_velocity module."""

from __future__ import annotations

from vlearn.torch_utils.torch_jit_utils import quat_rotate

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module
from optimal_morphology_rl.modules.robot_control.helpers import (
    build_action_space,
    validate_action_buffers,
    validate_robot_dependencies,
)


@register_module("robot_control_floating_hand")
class RobotControlFloatingHandModule(BaseModule):
    """Computes floating-hand base-velocity controls for the apply_root_velocity module.

    Reads the scaled policy actions from ``container.scaled_act_buf`` for the
    root slice and writes the current root transform plus the local base
    velocities (rotated into the world frame) into the ``set_root_*`` buffers
    that ``apply_root_velocity`` applies via
    ``gym.set_articulation_kinematic_states``.
    """

    def finalize(self, container: ModuleContainer) -> None:
        """Build the active motor mask and set the environment action space."""
        validate_robot_dependencies(container, "robot_control_floating_hand")
        if container.robot.fixed_hand:
            raise RuntimeError("robot_control_floating_hand is only supported for floating (non-fixed) hands.")
        build_action_space(container)

    def post_finalize(self, container: ModuleContainer) -> None:
        """Validate the action buffers are allocated."""
        validate_action_buffers(container, "robot_control_floating_hand")

    def step(self, container: ModuleContainer) -> None:
        """Write the root transform and world-frame base velocities."""
        robot = container.robot
        container.set_root_transform_buf[:] = robot.get_root_transform_buf
        local_root_vel = container.scaled_act_buf[:, container.root_slice]
        quat_robot_to_world = robot.get_root_transform_buf[:, 0:4]
        container.set_root_vel_buf[:, :3] = quat_rotate(quat_robot_to_world, local_root_vel[:, :3])
        container.set_root_vel_buf[:, 3:] = quat_rotate(quat_robot_to_world, local_root_vel[:, 3:])

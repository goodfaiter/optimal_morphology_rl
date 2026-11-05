"""Fixes the desired spool angle once per control step (the spool's hold: target)."""

from __future__ import annotations

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("update_spool_target")
class UpdateSpoolTargetModule(BaseModule):
    """With this module the spools of 'create_spool' hold a target (``spool.hold = "target"``; without it: offset):
    once per control step the desired spool angle is fixed at the current spool angle plus the policy's action, and
    for the physics substeps of that control step the servo law ('apply_m5_forces') moves the spool to it
    (``spool.commanded_delta_angle`` = target - the current angle). The current angle is read from the robot's joint
    buffers ('update_robot': after the last physics substep, or the reset state). List it in
    ``pre_physics_step_modules`` after 'process_variable_actions'::

        modules:
          pre_physics_step_modules: [process_variable_actions, update_spool_target]
    """

    def finalize(self, container: ModuleContainer) -> None:
        if container.get("spool") is None:
            raise RuntimeError("update_spool_target requires 'create_spool' (container.spool) in init_modules.")
        container.spool.hold = "target"

    def step(self, container: ModuleContainer) -> None:
        spool = container.spool
        angle = container.robot.get_joint_pos_buf[:, spool.dof_indices] / spool.pully_radius
        spool.target = angle + spool.action_delta_angle(container)

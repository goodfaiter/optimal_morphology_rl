"""Reads the servo spool state of 'create_spool' from the simulator."""

from __future__ import annotations

import torch

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("update_spool")
class UpdateSpoolModule(BaseModule):
    """Fills the spool state of 'create_spool' (``container.spool``) every step, per model tendon:

        theta_m = q / r,  omega_m = qdot / r            (winding positive; qdot the finite difference of q,
                                                          see CreateSpoolModule.slider_velocity)
        tau_e   = -r * T                                 (T: measured tendon tension, previous step)

    On the step after a reset the state uses the commanded reset joint state and zero tension. With ``hold: target``
    ('create_spool') the desired spool angle is fixed on a control step's first substep: the angle read then plus the
    policy's action. It runs every physics step: list it before the force law that reads the state (e.g.
    'apply_m5_forces') in ``pre_gym_step_modules`` (with 'update_robot' in ``post_gym_step_modules``, so every
    substep reads the state the last one reached); with ``frame_skip: 1`` ``pre_physics_step_modules`` works too::

        modules:
          init_modules: [..., create_spool, ...]
          pre_physics_step_modules: [process_variable_actions]
          pre_gym_step_modules: [update_spool, apply_m5_forces, ...]
          post_gym_step_modules: [update_robot]
    """

    def finalize(self, container: ModuleContainer) -> None:
        if container.get("spool") is None:
            raise RuntimeError("UpdateSpool requires 'create_spool' (container.spool) in init_modules.")

    def reset(self, container: ModuleContainer) -> None:
        """On the next step use the commanded reset state (and zero tension) for the reset envs."""
        container.spool.just_reset |= container.reset_buf

    def step(self, container: ModuleContainer) -> None:
        """Read the spool state of every model tendon from the simulator."""
        substep = container.get("substep")
        if substep is None and int(container.get("frame_skip", 1)) > 1:
            raise RuntimeError("update_spool reads the spool every physics step: with frame_skip > 1 list it in "
                               "pre_gym_step_modules (and update_robot in post_gym_step_modules).")
        spool, robot = container.spool, container.robot
        r = spool.pully_radius

        slider_pos = robot.get_joint_pos_buf[:, spool.dof_indices]
        slider_vel = spool.slider_velocity(container)
        tension = robot.get_tendon_force_buf[:, spool.model_tendon_indices].clamp(min=0.0)

        spool.reset_rows[:] = spool.just_reset
        if bool(torch.any(spool.just_reset)):
            just_reset = spool.just_reset.unsqueeze(-1)
            slider_pos = torch.where(just_reset, container.reset_joint_pos_buf[:, spool.dof_indices], slider_pos)
            tension = torch.where(just_reset, torch.zeros_like(tension), tension)
            spool.just_reset[:] = False

        spool.last_slider_pos[:] = slider_pos
        # New tensors every step, so references kept by readers (e.g. last_signals) stay valid.
        spool.theta_m = slider_pos / r
        spool.omega_m = slider_vel / r
        spool.tension = tension
        spool.tau_e = -r * tension
        if spool.hold == "target" and not substep:  # a new action: fix the desired spool angle
            spool.target = spool.theta_m + spool.action_delta_angle(container)

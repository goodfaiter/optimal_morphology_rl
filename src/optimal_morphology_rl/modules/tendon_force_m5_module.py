"""Analytic virtual-motor tendon model producing tendon forces (alternative to 'tendon_est')."""

from __future__ import annotations

import torch

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module
from optimal_morphology_rl.modules.tendon_scale_utils import build_tendon_scale_tensor


@register_module("tendon_force_m5")
class TendonForceM5Module(BaseModule):
    """Analytic virtual-motor tendon model producing tendon forces.

    A virtual motor with a spool of radius ``pully_radius`` drives each model
    tendon. The tendon velocity/acceleration inputs are the filtered
    finite-difference signals ``calculated_tendon_velocity`` and
    ``calculated_tendon_acceleration`` computed by 'update_robot' (see there
    for the finite differencing, spike guard, exponential smoothing and soft
    deadbands). The spool converts between tendon and motor units via the
    winding convention ``dL = -radius * dtheta`` (as in 'tendon_est'):

    - ``omega``: motor rate, the filtered tendon velocity divided by
      ``pully_radius``
    - ``acc``: motor acceleration, the filtered tendon acceleration divided by
      ``pully_radius``
    - ``tau_m``: motor torque ``motor_gain * action``
    - ``tau_e_prev``: transmitted torque from the previous spatial tendon force
      read from the simulator (``-force * radius``, see 'update_robot')

    The joint equation of motion is ``tau_m + tau_e + tau_f = J * acc``. The
    friction ``tau_f`` is the implicit stop-torque ``-(J/dt * omega + tau_m +
    tau_e_prev)`` capped by the Stribeck friction limit::

        tau_f_max = K_v*|omega| + K_c + |K_m*tau_m - K_e*tau_e_prev|
                  + exp(-|omega/v_s|^alpha) * (K_cs + |K_ms*tau_m - K_es*tau_e_prev|)

    The transmitted torque is then ``tau_e = J * acc - tau_m - tau_f``. The
    vsim convention is that ``tau_e`` and the spatial tendon force are opposite
    in sign, so the applied force is ``-tau_e / radius`` (clamped non-negative
    — tendons only pull).

    ``pully_radius`` applies to the model-driven tendons only (scalar, or one
    entry per ``model_tendon_indices`` entry) — tendons excluded from the model
    (e.g. a rigid DIP) need no entries. All model coefficients are scalars, or
    one entry per model tendon.

    The computed force is written to ``container.tendon_force_buf`` for the
    model-driven tendon columns (see ``container.tendon_model_indices``) so
    ``robot_control_tendons`` can apply it via ``set_spatial_tendon_forces``.
    This module is an alternative to 'tendon_est': do not list both (both
    write ``container.tendon_force_buf``).
    """

    def finalize(self, container: ModuleContainer) -> None:
        """Validate cross-module dependencies and the model-driven tendon config."""
        if container.get("robot") is None:
            raise RuntimeError(
                "TendonForceM5 requires 'robot' in the shared container. "
                "Ensure the 'create_robot' module is listed before 'tendon_force_m5'."
            )
        if container.get("env") is None:
            raise RuntimeError("TendonForceM5 requires 'env' in the shared container.")

        robot = container.robot
        if not robot.use_tendon:
            raise RuntimeError(
                "TendonForceM5 only supports tendon-driven robot configurations (the 'create_robot' config must set 'use_tendon: true')."
            )

        indices = self.config.get("model_tendon_indices")
        if not isinstance(indices, (list, tuple)) or len(indices) == 0:
            raise RuntimeError("TendonForceM5 config missing 'model_tendon_indices': the list of tendon columns the model drives.")
        indices = list(indices)
        for idx in indices:
            if not isinstance(idx, int) or not (0 <= idx < robot.num_tendons):
                raise RuntimeError(
                    f"TendonForceM5 config 'model_tendon_indices' entries must be tendon indices in [0, {robot.num_tendons}), got {idx}."
                )
        self.model_tendon_indices = indices

    def post_finalize(self, container: ModuleContainer) -> None:
        """Allocate the force buffer and build the model coefficient tensors."""
        robot = container.robot
        device = container.device
        num_model_tendons = len(self.model_tendon_indices)

        timestep = container.get("timestep")
        frame_skip = container.get("frame_skip")
        if timestep is None or frame_skip is None:
            raise RuntimeError(
                "TendonForceM5 requires 'timestep' and 'frame_skip' in the shared container. "
                "Ensure the 'create_rigid_vsim_envs' module is listed before 'tendon_force_m5'."
            )
        self.dt = float(timestep) * float(frame_skip)

        container.tendon_force_buf = torch.zeros((container.total_num_envs, robot.num_tendons), device=device, dtype=torch.float32)
        container.tendon_model_indices = torch.tensor(self.model_tendon_indices, device=device, dtype=torch.long)

        self.pully_radius = build_tendon_scale_tensor(self.config.get("pully_radius", 0.011), num_model_tendons, device, "pully_radius")
        self.motor_gain = build_tendon_scale_tensor(self.config.get("motor_gain", 0.105), num_model_tendons, device, "motor_gain")
        self.inertia = build_tendon_scale_tensor(self.config.get("J", 0.0011779299150459918), num_model_tendons, device, "J")
        self.k_v = build_tendon_scale_tensor(self.config.get("K_v", 0.0037641), num_model_tendons, device, "K_v")
        self.k_c = build_tendon_scale_tensor(self.config.get("K_c", 0.00666241), num_model_tendons, device, "K_c")
        self.k_m = build_tendon_scale_tensor(self.config.get("K_m", 0.447678), num_model_tendons, device, "K_m")
        self.k_e = build_tendon_scale_tensor(self.config.get("K_e", 0.00129603), num_model_tendons, device, "K_e")
        self.v_s = build_tendon_scale_tensor(self.config.get("v_s", 0.0628076), num_model_tendons, device, "v_s")
        self.alpha = build_tendon_scale_tensor(self.config.get("alpha", 1.00722), num_model_tendons, device, "alpha")
        self.k_cs = build_tendon_scale_tensor(self.config.get("K_cs", 0.000773996), num_model_tendons, device, "K_cs")
        self.k_ms = build_tendon_scale_tensor(self.config.get("K_ms", 0.303106), num_model_tendons, device, "K_ms")
        self.k_es = build_tendon_scale_tensor(self.config.get("K_es", 0.00117), num_model_tendons, device, "K_es")

        if torch.any(self.pully_radius <= 0.0):
            raise RuntimeError("TendonForceM5 config 'pully_radius' entries must be positive.")
        if torch.any(self.v_s <= 0.0):
            raise RuntimeError("TendonForceM5 config 'v_s' entries must be positive.")

    def step(self, container: ModuleContainer) -> None:
        """Compute the virtual-motor tendon force for every model tendon."""
        robot = container.robot

        # Filtered tendon velocity/acceleration computed by 'update_robot' (post-physics).
        tendon_vels = robot.calculated_tendon_velocity[:, self.model_tendon_indices]
        acc = robot.calculated_tendon_acceleration[:, self.model_tendon_indices] / self.pully_radius

        sim_forces = -1.0 * robot.get_tendon_force_buf[:, self.model_tendon_indices]
        actions = container.scaled_act_buf[:, container.active_dof_slice]

        # Map each model tendon to its policy-action slot within the
        # policy-controlled tendon columns (fixed tendons have no action).
        dof_indices = [int(v) for v in container.active_dof_indices.tolist()]
        action_slots = [dof_indices.index(tendon_idx) for tendon_idx in self.model_tendon_indices]

        # dL = -radius * dtheta (winding convention, as in 'tendon_est').
        omega = tendon_vels / self.pully_radius

        # Motor torque: PD gain on (action - measured motor angle).
        tau_m = self.motor_gain * (actions[:, action_slots])

        # Transmitted torque from the previous spatial tendon force read from the simulator.
        tau_e_prev = sim_forces * self.pully_radius

        # Stribeck friction limit: viscous + Coulomb + decaying static term.
        coulomb = self.k_c + (self.k_m * tau_m - self.k_e * tau_e_prev).abs()
        static = torch.exp(-((omega.abs() / self.v_s) ** self.alpha)) * (self.k_cs + (self.k_ms * tau_m - self.k_es * tau_e_prev).abs())
        tau_f_max = self.k_v * omega.abs() + coulomb + static

        # Friction is the implicit stop-torque capped by the Stribeck limit.
        tau_f_stop = -(self.inertia / self.dt * omega + tau_m + tau_e_prev)
        tau_f = torch.clamp(tau_f_stop, -tau_f_max, tau_f_max)

        # Environment/transmitted torque from the equation of motion.
        inertia_force = self.inertia * acc
        tau_e = inertia_force - tau_m - tau_f

        # the convention in vsim is that teu_e and tendon force are opposite in sign, so we negate tau_e to get the applied tendon force
        force = torch.clamp(-1.0 * tau_e / self.pully_radius, min=0.0)

        container.tendon_force_buf[:, self.model_tendon_indices] = force

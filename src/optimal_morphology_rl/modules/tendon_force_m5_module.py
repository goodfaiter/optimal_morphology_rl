"""Analytic virtual-motor tendon model producing tendon forces (alternative to 'tendon_est')."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


def _build_tendon_scale_tensor(cfg_value: Any, expected_len: int, device: torch.device, name: str) -> torch.Tensor:
    """Convert a scalar or per-model-tendon list config value to a float tensor."""
    if isinstance(cfg_value, (list, tuple)):
        if len(cfg_value) != expected_len:
            raise RuntimeError(f"TendonForceM5 config '{name}' length ({len(cfg_value)}) must match {expected_len}.")
        return torch.tensor(cfg_value, device=device, dtype=torch.float32)
    return torch.full((expected_len,), float(cfg_value), device=device, dtype=torch.float32)


@register_module("tendon_force_m5")
class TendonForceM5Module(BaseModule):
    """Analytic virtual-motor tendon model producing tendon forces.

    A virtual motor with a spool of radius ``pully_radius`` drives each model
    tendon. The spool converts between tendon length/velocity and motor
    angle/rate via the winding convention ``dL = -radius * dtheta`` (as in
    'tendon_est'):

    - ``omega``: motor rate from the finite-difference tendon speed
      (``-(L - L_prev)/dt`` per step, winding convention — the simulator's
      tendon velocity buffer is unreliable and unused), exponentially smoothed
      (``vel = a*vel_raw + (1-a)*vel_prev`` with the ``vel_filter_alpha``
      coefficient, 1.0 = raw), spike-guarded (``max_tendon_speed`` — on the
      reset step the buffers still show the pre-teleport state, so the next
      step would otherwise see a phantom velocity jump that launches the
      finger) and soft-debanded (``vel_deadband``, attenuates
      the residual length-jitter noise floor so the acceleration FD cannot
      amplify it into bang-bang force chatter at near-zero velocity)
    - ``acc``: motor acceleration from the finite difference of the speed,
      exponentially smoothed (``acc = a*acc_raw + (1-a)*acc_prev`` with the
      ``acc_filter_alpha`` coefficient, 1.0 = raw) and soft-debanded
      (``acc_deadband``)
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

        self.pully_radius = _build_tendon_scale_tensor(self.config.get("pully_radius", 0.011), num_model_tendons, device, "pully_radius")
        self.max_tendon_speed = _build_tendon_scale_tensor(
            self.config.get("max_tendon_speed", 0.22), num_model_tendons, device, "max_tendon_speed"
        )

        self.motor_gain = _build_tendon_scale_tensor(self.config.get("motor_gain", 0.105), num_model_tendons, device, "motor_gain")
        self.inertia = _build_tendon_scale_tensor(self.config.get("J", 0.0011779299150459918), num_model_tendons, device, "J")
        self.k_v = _build_tendon_scale_tensor(self.config.get("K_v", 0.0037641), num_model_tendons, device, "K_v")
        self.k_c = _build_tendon_scale_tensor(self.config.get("K_c", 0.00666241), num_model_tendons, device, "K_c")
        self.k_m = _build_tendon_scale_tensor(self.config.get("K_m", 0.447678), num_model_tendons, device, "K_m")
        self.k_e = _build_tendon_scale_tensor(self.config.get("K_e", 0.00129603), num_model_tendons, device, "K_e")
        self.v_s = _build_tendon_scale_tensor(self.config.get("v_s", 0.0628076), num_model_tendons, device, "v_s")
        self.alpha = _build_tendon_scale_tensor(self.config.get("alpha", 1.00722), num_model_tendons, device, "alpha")
        self.k_cs = _build_tendon_scale_tensor(self.config.get("K_cs", 0.000773996), num_model_tendons, device, "K_cs")
        self.k_ms = _build_tendon_scale_tensor(self.config.get("K_ms", 0.303106), num_model_tendons, device, "K_ms")
        self.k_es = _build_tendon_scale_tensor(self.config.get("K_es", 0.00117), num_model_tendons, device, "K_es")
        self.acc_filter_alpha = _build_tendon_scale_tensor(
            self.config.get("acc_filter_alpha", 1.0), num_model_tendons, device, "acc_filter_alpha"
        )
        self.vel_filter_alpha = _build_tendon_scale_tensor(
            self.config.get("vel_filter_alpha", 1.0), num_model_tendons, device, "vel_filter_alpha"
        )
        self.vel_deadband = _build_tendon_scale_tensor(self.config.get("vel_deadband", 0.0), num_model_tendons, device, "vel_deadband")
        self.acc_deadband = _build_tendon_scale_tensor(self.config.get("acc_deadband", 0.0), num_model_tendons, device, "acc_deadband")

        if torch.any(self.pully_radius <= 0.0):
            raise RuntimeError("TendonForceM5 config 'pully_radius' entries must be positive.")
        if torch.any(self.max_tendon_speed <= 0.0):
            raise RuntimeError("TendonForceM5 config 'max_tendon_speed' entries must be positive.")
        if torch.any(self.v_s <= 0.0):
            raise RuntimeError("TendonForceM5 config 'v_s' entries must be positive.")
        if torch.any(self.acc_filter_alpha < 0.0) or torch.any(self.acc_filter_alpha > 1.0):
            raise RuntimeError("TendonForceM5 config 'acc_filter_alpha' entries must be in [0, 1].")
        if torch.any(self.vel_filter_alpha < 0.0) or torch.any(self.vel_filter_alpha > 1.0):
            raise RuntimeError("TendonForceM5 config 'vel_filter_alpha' entries must be in [0, 1].")
        if torch.any(self.vel_deadband < 0.0):
            raise RuntimeError("TendonForceM5 config 'vel_deadband' entries must be non-negative.")
        if torch.any(self.acc_deadband < 0.0):
            raise RuntimeError("TendonForceM5 config 'acc_deadband' entries must be non-negative.")

        # Previous filtered tendon speed per env/model tendon (exponential filter state,
        # also the previous sample for the acceleration finite difference).
        self.prev_vel = torch.zeros((container.total_num_envs, num_model_tendons), device=device, dtype=torch.float32)
        # Previous tendon length per env/model tendon for the velocity finite difference.
        # NaN = no history yet: the first step after (re)initialization uses zero velocity.
        self.prev_length = torch.full((container.total_num_envs, num_model_tendons), float("nan"), device=device, dtype=torch.float32)
        # Previous filtered acceleration per env/model tendon (exponential filter state).
        self.prev_acc = torch.zeros((container.total_num_envs, num_model_tendons), device=device, dtype=torch.float32)

    def step(self, container: ModuleContainer) -> None:
        """Compute the virtual-motor tendon force for every model tendon."""
        robot = container.robot

        tendon_lengths = robot.get_tendon_lengths_buf[:, self.model_tendon_indices]

        # Finite-difference tendon velocity (the simulator's velocity buffer is unreliable).
        # Entries without length history (first step after reset) use zero velocity.
        has_history = torch.isfinite(self.prev_length)
        vel_raw = -1.0 * (tendon_lengths - self.prev_length) / self.dt
        vel_raw = torch.where(has_history, vel_raw, torch.zeros_like(vel_raw))

        # Spike guard: on the reset step the buffers still show the pre-teleport state,
        # so the first stored length is stale and the next step sees a phantom velocity
        # jump far beyond the physical joint speed limit (which launches the finger).
        # Jumps beyond max_tendon_speed are phantom velocities, not motion.
        vel_raw = torch.where(vel_raw.abs() > self.max_tendon_speed, torch.zeros_like(vel_raw), vel_raw)

        # Exponential smoothing of the noisy finite-difference velocity.
        tendon_vels = self.vel_filter_alpha * vel_raw + (1.0 - self.vel_filter_alpha) * self.prev_vel

        # Soft deadband (continuous, so no FD spikes at the threshold): attenuates the
        # residual length-jitter noise floor so the acceleration finite difference does
        # not amplify it into alternating accelerations (bang-bang force chatter at
        # near-zero velocity).
        tendon_vels = torch.sign(tendon_vels) * torch.clamp(tendon_vels.abs() - self.vel_deadband, min=0.0)

        sim_forces = -1.0 * robot.get_tendon_force_buf[:, self.model_tendon_indices]
        actions = container.scaled_act_buf[:, container.active_dof_slice]

        # Map each model tendon to its policy-action slot within the
        # policy-controlled tendon columns (fixed tendons have no action).
        dof_indices = [int(v) for v in container.active_dof_indices.tolist()]
        action_slots = [dof_indices.index(tendon_idx) for tendon_idx in self.model_tendon_indices]

        # dL = -radius * dtheta (winding convention, as in 'tendon_est').
        omega = tendon_vels / self.pully_radius
        omega_prev = self.prev_vel / self.pully_radius
        acc_raw = (omega - omega_prev) / self.dt

        # Exponential smoothing of the noisy finite-difference acceleration.
        acc = self.acc_filter_alpha * acc_raw + (1.0 - self.acc_filter_alpha) * self.prev_acc

        # Soft deadband on the acceleration (same rationale as the velocity deadband).
        acc = torch.sign(acc) * torch.clamp(acc.abs() - self.acc_deadband, min=0.0)

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
        self.prev_length[:] = tendon_lengths
        self.prev_vel[:] = tendon_vels
        self.prev_acc[:] = acc

    def reset(self, container: ModuleContainer) -> None:
        """Clear the length/velocity/acceleration history of the environments selected by the reset buffer."""
        self.prev_length[container.reset_buf] = float("nan")
        self.prev_vel[container.reset_buf] = 0.0
        self.prev_acc[container.reset_buf] = 0.0

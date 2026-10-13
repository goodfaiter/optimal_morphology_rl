"""Servo-motor tendon model (M5 friction) producing tendon forces (alternative to 'tendon_est')."""

from __future__ import annotations

import torch

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module
from optimal_morphology_rl.modules.tendon_scale_utils import build_tendon_scale_tensor


@register_module("tendon_force_m5")
class TendonForceM5Module(BaseModule):
    """Servo motor with its own state, coupled to the motor end of each model tendon by a spring.

    The simulator does not know about the motor's inertia, so the motor is integrated here:
    J (already reflected, J_rotor * N**2) lives in this module's state. The coupling spring
    connects that state to the sim tendon; it is a numerical coupling, so K_coupling should be
    as stiff as stability allows (see below), not a model of cable elasticity.

        F_k   = K * r * (theta_m - theta_t) + C * r * (omega_m - omega_t),  only while stretched
        T_k   = T_sim + (F_k - F_0)            (measured tension + in-step coupling correction)
        tau_e = -r * T_k
        J * domega_m = tau_m + tau_f + tau_e,  tau_m = motor_gain * (action - theta_m)
        applied force = mean of F_k over the substeps

    Friction: M5 (directional, Stribeck), Karnopp stop torque capped by the friction budget.

    Stability. The motor side is integrated with substeps and is not the limit. The finger side is:
    the applied force is held constant for the whole control step dt while the (light) finger
    reacts, an explicit coupling. With m_f the finger's effective mass at the tendon, roughly
        K_coupling < 4 * m_f / dt**2        and        C_coupling < 2 * m_f / dt
    Above these, the finger chatters at the sampling rate (period-2 oscillation).
    """

    def finalize(self, container: ModuleContainer) -> None:
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
        robot = container.robot
        device = container.device
        num_envs = container.total_num_envs
        num_model_tendons = len(self.model_tendon_indices)

        timestep = container.get("timestep")
        frame_skip = container.get("frame_skip")
        if timestep is None or frame_skip is None:
            raise RuntimeError(
                "TendonForceM5 requires 'timestep' and 'frame_skip' in the shared container. "
                "Ensure the 'create_rigid_vsim_envs' module is listed before 'tendon_force_m5'."
            )
        self.dt = float(timestep) * float(frame_skip)

        container.tendon_force_buf = torch.zeros((num_envs, robot.num_tendons), device=device, dtype=torch.float32)
        container.tendon_model_indices = torch.tensor(self.model_tendon_indices, device=device, dtype=torch.long)

        def coef(name, default):
            return build_tendon_scale_tensor(self.config.get(name, default), num_model_tendons, device, name)

        self.pully_radius = coef("pully_radius", 0.011)
        self.motor_gain = coef("motor_gain", 0.105)
        self.inertia = coef("J", 0.0011779299150459918)  # reflected: J_rotor * N**2
        self.k_v = coef("K_v", 0.0037641)
        self.k_c = coef("K_c", 0.00666241)
        self.k_m = coef("K_m", 0.447678)
        self.k_e = coef("K_e", 0.00129603)
        self.v_s = coef("v_s", 0.0628076)
        self.alpha = coef("alpha", 1.00722)
        self.k_cs = coef("K_cs", 0.000773996)
        self.k_ms = coef("K_ms", 0.303106)
        self.k_es = coef("K_es", 0.00117)
        self.tendon_offsets = coef("tendon_offsets", [0.0313438, 0.03986271])

        # Coupling spring between the motor state and the sim tendon (N/m, N*s/m at the tendon).
        self.k_coupling = coef("K_coupling", 20000.0)
        self.c_coupling = coef("C_coupling", 80.0)
        self.n_substeps = int(self.config.get("n_substeps", 10))

        # Optional debug print of one (env, tendon) pair. Off by default: .cpu() forces a GPU sync.
        self.debug = bool(self.config.get("debug", False))
        self.debug_env = int(self.config.get("debug_env", 0))
        self.debug_tendon = int(self.config.get("debug_tendon", 0))

        if torch.any(self.pully_radius <= 0.0):
            raise RuntimeError("TendonForceM5 config 'pully_radius' entries must be positive.")
        if torch.any(self.v_s <= 0.0):
            raise RuntimeError("TendonForceM5 config 'v_s' entries must be positive.")
        if torch.any(self.inertia <= 0.0):
            raise RuntimeError("TendonForceM5 config 'J' entries must be positive.")
        if torch.any(self.k_coupling <= 0.0) or torch.any(self.c_coupling < 0.0):
            raise RuntimeError("TendonForceM5 config 'K_coupling' must be positive and 'C_coupling' non-negative.")
        if self.n_substeps < 1:
            raise RuntimeError("TendonForceM5 config 'n_substeps' must be >= 1.")

        self.theta_m = torch.zeros((num_envs, num_model_tendons), device=device, dtype=torch.float32)
        self.omega_m = torch.zeros_like(self.theta_m)
        self.needs_sync = torch.ones(num_envs, device=device, dtype=torch.bool)
        self.action_slots = None

    def reset(self, container: ModuleContainer, env_ids=None) -> None:
        """Re-sync the motor state to the tendon on the next step."""
        if env_ids is None:
            self.needs_sync[:] = True
        else:
            self.needs_sync[env_ids] = True

    def _coupling_force(self, theta_m, omega_m, theta_t, omega_t):
        """Tension-only spring + damper; no force at all while slack."""
        r = self.pully_radius
        stretch = r * (theta_m - theta_t)
        f = self.k_coupling * stretch + self.c_coupling * r * (omega_m - omega_t)
        return torch.where(stretch > 0.0, f.clamp(min=0.0), torch.zeros_like(f))

    def _friction_budget(self, tau_m, tau_e, w_abs):
        """M5: directional load-dependent friction with Stribeck decay (paper, eq. 9)."""
        coulomb = self.k_c + (self.k_m * tau_m - self.k_e * tau_e).abs()
        static = torch.exp(-((w_abs / self.v_s) ** self.alpha)) * (self.k_cs + (self.k_ms * tau_m - self.k_es * tau_e).abs())
        return self.k_v * w_abs + coulomb + static

    def step(self, container: ModuleContainer) -> None:
        robot = container.robot
        r = self.pully_radius
        J = self.inertia

        if self.action_slots is None:
            dof_indices = [int(v) for v in container.active_dof_indices.tolist()]
            self.action_slots = torch.tensor(
                [dof_indices.index(tendon_idx) for tendon_idx in self.model_tendon_indices],
                device=container.device,
                dtype=torch.long,
            )

        # Motor end of the sim tendon. Winding positive: theta increases as the tendon shortens.
        lengths = robot.get_tendon_lengths_buf[:, self.model_tendon_indices]
        theta_t = (self.tendon_offsets - lengths) / r
        omega_t = robot.calculated_tendon_velocity[:, self.model_tendon_indices] / r

        just_synced = self.needs_sync.clone()
        if torch.any(just_synced):
            self.theta_m[just_synced] = theta_t[just_synced]
            self.omega_m[just_synced] = omega_t[just_synced]
            self.needs_sync[:] = False

        # Tension from the previous step (positive = pulling). Stale on the first step after a reset.
        f_sim = robot.get_tendon_force_buf[:, self.model_tendon_indices].clamp(min=0.0)
        f_sim = torch.where(just_synced.unsqueeze(-1), torch.zeros_like(f_sim), f_sim)

        actions = container.scaled_act_buf[:, container.active_dof_slice][:, self.action_slots]

        h = self.dt / self.n_substeps
        theta_m = self.theta_m
        omega_m = self.omega_m
        force_sum = torch.zeros_like(theta_m)

        # Coupling force at the start of the step: reference for the in-step correction.
        f0 = self._coupling_force(theta_m, omega_m, theta_t, omega_t)

        for k in range(self.n_substeps):
            theta_t_k = theta_t + omega_t * (k * h)
            force = self._coupling_force(theta_m, omega_m, theta_t_k, omega_t)

            # Measured tension, corrected by how the coupling has changed since the step began.
            tension = (f_sim + (force - f0)).clamp(min=0.0)
            tau_e = -r * tension
            tau_m = self.motor_gain * (actions - theta_m)

            w_abs = omega_m.abs()
            tau_f_max = self._friction_budget(tau_m, tau_e, w_abs)
            tau_f_stop = -(J / h * omega_m + tau_m + tau_e)
            tau_f = torch.clamp(tau_f_stop, -tau_f_max, tau_f_max)

            omega_m = omega_m + h * (tau_m + tau_f + tau_e) / J
            theta_m = theta_m + h * omega_m
            force_sum = force_sum + force

        self.theta_m = theta_m
        self.omega_m = omega_m
        force = force_sum / self.n_substeps

        # if self.debug:
        e, t = 0, 1
        vals = {
            "theta_m": theta_m, "theta_t": theta_t, "omega_m": omega_m, "omega_t": omega_t,
            "stretch": r * (theta_m - theta_t), "f_sim": f_sim, "tau_e": tau_e,
            "tau_m": tau_m, "tau_f": tau_f, "force": force,
        }
        print(" ".join(f"{name} {v[e, t].item(): .4f}" for name, v in vals.items()))

        container.tendon_force_buf[:, self.model_tendon_indices] = force
"""M5-friction servo law on the 'create_spool' spools: computes the slider motor forces."""

from __future__ import annotations

from pathlib import Path

import optimal_morphology_rl_assets
import torch

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module
from optimal_morphology_rl.modules.tendon_scale_utils import build_tendon_scale_tensor

# Config key -> attribute of the servo law parameters (each (num_envs, num_model_tendons)).
M5_PARAMS = {
    "motor_position_gain": "motor_position_gain",
    "torque_constant": "torque_constant",
    "current_deadband": "current_deadband",
    "K_v": "k_v",
    "K_c": "k_c",
    "K_m": "k_m",
    "K_e_wind": "k_e_wind",
    "K_e_unwind": "k_e_unwind",
    "v_s": "v_s",
    "alpha": "alpha",
    "K_cs": "k_cs",
    "K_ms": "k_ms",
    "K_es": "k_es",
}
# Spool speed (rad/s) below which the friction direction is the one the spool is pushed in (tau_m + tau_e).
WIND_DIRECTION_EPS = 1e-3


def _resolve_model_path(path: str) -> Path:
    """Absolute paths are used as is, relative ones resolve against the assets 'models/actuators' folder."""
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(optimal_morphology_rl_assets.__file__).resolve().parent / "models" / "actuators" / p


@register_module("apply_m5_forces")
class ApplyM5ForcesModule(BaseModule):
    """Servo motor law with M5 friction on the spools of 'create_spool' (``container.spool``).

    From the spool state (theta_m, omega_m, tau_e) and the policy action, the desired spool angle (rad):

        target  = max(action, 0)                         (never below the tared taut angle, as on the robot)
        i       = motor_position_gain * (target - theta_m)            (the servo's current command, A)
        tau_m   = torque_constant * sign(i) * max(|i| - current_deadband, 0)   (motor torque, Nm)
        tau_f   = M5 friction, Karnopp stop torque capped by the friction budget; the load coefficient K_e is
                  K_e_wind while the spool winds (theta increasing, tendon shortening) and K_e_unwind while it
                  unwinds (at standstill: the direction tau_m + tau_e pushes it)
        F       = (tau_m + tau_f) / r                    (motor force on the slider DOF)

    The tendon pulls the slider with -T and the servo inertia is the slider's armature, so the
    simulator closes the loop  J_eff * domega_m = tau_m + tau_f + tau_e  with the spool's J_eff.

    With 'max_slider_velocity' (m/s) the servo speed is limited (vlearn does not enforce the slider
    joint's velocity limit): tau_m + tau_f is capped so that this loop keeps |omega_m| within
    max_slider_velocity / r on the next step. The limit is per env and tendon
    ``(num_envs, num_model_tendons)`` so it can be randomized ('randomize_slider_velocity'); the
    configured value is kept in ``nominal_max_slider_velocity``.

    With 'friction_model_path' set, a learned TorchScript model replaces the M5 constants:
    ``model(omega, theta_delta)`` with the spool speed (rad/s) and theta_delta = target - theta_m (rad), at
    the 200 Hz control rate. ``friction_model_output`` says what it predicts: ``envelope`` (default) the
    friction budget tau_f_max (the Karnopp stop torque still uses the simulator's J_eff), ``torque`` the
    signed friction torque tau_f itself (applied as is). Its output is multiplied by the constant
    ``friction_model_gain`` (default 1.0) and by ``friction_model_scale`` ``(num_envs,
    num_model_tendons)`` (ones; randomized by 'randomize_m5'). The model is stateful (input history per row): it runs one forward pass over
    num_envs x num_model_tendons rows, and ``reset`` restarts the rows of the resetting envs.

    With ``friction: false`` the friction torque is left out (tau_f = 0, F = tau_m / r): an ideal,
    frictionless servo, e.g. to compare against the M5 law on the robot.

    The parameters (the motor constants and the M5 coefficients, see ``M5_PARAMS``) are per env and
    tendon ``(num_envs, num_model_tendons)`` so they can be randomized ('randomize_m5'); the
    configured values are kept in ``nominal``.

    The forces go to ``container.spool_motor_force_buf`` and are applied by 'apply_motor_forces'.
    List it in ``pre_physics_step_modules`` after 'create_spool'.
    """

    def finalize(self, container: ModuleContainer) -> None:
        if container.get("spool") is None:
            raise RuntimeError("ApplyM5Forces requires 'create_spool' (container.spool) to be listed before 'apply_m5_forces'.")

    def post_finalize(self, container: ModuleContainer) -> None:
        spool = container.spool
        device = container.device
        num_tendons = len(spool.model_tendon_indices)
        self.dt = spool.dt

        def coef(name, default):
            return build_tendon_scale_tensor(self.config.get(name, default), num_tendons, device, name)

        self.motor_position_gain = coef("motor_position_gain", 0.1)  # A/rad, as on the robot
        self.torque_constant = coef("torque_constant", 0.48)  # Nm/A
        self.current_deadband = coef("current_deadband", 0.042)  # A, current that makes no torque
        self.k_v = coef("K_v", 0.0037641)
        self.k_c = coef("K_c", 0.00666241)
        self.k_m = coef("K_m", 0.447678)
        self.k_e_wind = coef("K_e_wind", 0.00129603)
        self.k_e_unwind = coef("K_e_unwind", 0.00129603)
        self.v_s = coef("v_s", 0.0628076)
        self.alpha = coef("alpha", 1.00722)
        self.k_cs = coef("K_cs", 0.000773996)
        self.k_ms = coef("K_ms", 0.303106)
        self.k_es = coef("K_es", 0.00117)
        if torch.any(self.v_s <= 0.0):
            raise RuntimeError("ApplyM5Forces config 'v_s' entries must be positive.")
        if torch.any(self.current_deadband < 0.0):
            raise RuntimeError("ApplyM5Forces config 'current_deadband' entries must be non-negative.")
        # Configured values per tendon; the attributes become per env so they can be randomized.
        num_envs = container.total_num_envs
        self.nominal = {key: getattr(self, attr).clone() for key, attr in M5_PARAMS.items()}
        for key, attr in M5_PARAMS.items():
            setattr(self, attr, self.nominal[key].expand(num_envs, num_tendons).clone())
        for key in ("pully_radius", "model_tendon_indices", "slider_joint_suffix", "velocity_source"):
            if key in self.config:
                raise RuntimeError(f"ApplyM5Forces: '{key}' belongs to the 'create_spool' (or 'update_robot') config.")

        # Optional learned friction envelope (TorchScript: model(omega, theta_delta) -> tau_f_max, Nm).
        self.friction_model = None
        model_path = self.config.get("friction_model_path")
        if model_path:
            model_path = _resolve_model_path(str(model_path))
            if not model_path.is_file():
                raise RuntimeError(f"ApplyM5Forces: friction model '{model_path}' not found.")
            self.friction_model = torch.jit.load(str(model_path), map_location=device)
            self.friction_model.eval()
            frequency = float(getattr(self.friction_model, "frequency", 1.0 / self.dt))
            if abs(frequency * self.dt - 1.0) > 1e-6:
                print(f"ApplyM5Forces: WARNING friction model runs at {frequency:g} Hz, the env control at {1.0 / self.dt:g} Hz")
            self.friction_model_output = str(self.config.get("friction_model_output", "envelope"))
            if self.friction_model_output not in ("envelope", "torque"):
                raise RuntimeError("ApplyM5Forces config 'friction_model_output' must be 'envelope' or 'torque'.")
            self.friction_model_gain = float(self.config.get("friction_model_gain", 1.0))
            if self.friction_model_gain < 0.0:
                raise RuntimeError("ApplyM5Forces config 'friction_model_gain' must be non-negative.")
            # Per env and tendon factor on the envelope (randomize_m5 draws it).
            self.friction_model_scale = torch.ones((num_envs, num_tendons), device=device, dtype=torch.float32)
            print(f"ApplyM5Forces: friction {self.friction_model_output} from '{model_path.name}' (gain {self.friction_model_gain:g})")

        container.spool_motor_force_buf = torch.zeros(
            (container.total_num_envs, container.robot.num_motors), device=device, dtype=torch.float32
        )

        max_v = self.config.get("max_slider_velocity")
        self.nominal_max_slider_velocity = None if max_v is None else float(max_v)
        self.max_slider_velocity = None
        if self.nominal_max_slider_velocity is not None:
            if self.nominal_max_slider_velocity <= 0.0:
                raise RuntimeError("ApplyM5Forces config 'max_slider_velocity' must be positive (m/s).")
            self.max_slider_velocity = torch.full(
                (num_envs, num_tendons), self.nominal_max_slider_velocity, device=device, dtype=torch.float32
            )

        self.friction = bool(self.config.get("friction", True))

        # Optional debug print of one (env, tendon) pair. Off by default: .item() forces a GPU sync.
        self.debug = bool(self.config.get("debug", False))
        self.debug_env = int(self.config.get("debug_env", 0))
        self.debug_tendon = int(self.config.get("debug_tendon", 0))

    def _motor_torque(self, error):
        """Motor torque of a spool angle error (target - theta): current command, then torque past the deadband."""
        current = self.motor_position_gain * error
        return self.torque_constant * torch.sign(current) * torch.clamp(current.abs() - self.current_deadband, min=0.0)

    def _friction_budget(self, tau_m, tau_e, omega):
        """M5: directional load-dependent friction with Stribeck decay (paper, eq. 9).

        omega: signed spool speed (rad/s, + = winding); K_e is K_e_wind / K_e_unwind by direction.
        """
        w_abs = omega.abs()
        winding = torch.where(w_abs > WIND_DIRECTION_EPS, omega > 0.0, tau_m + tau_e > 0.0)
        k_e = torch.where(winding, self.k_e_wind, self.k_e_unwind)
        coulomb = self.k_c + (self.k_m * tau_m - k_e * tau_e).abs()
        static = torch.exp(-((w_abs / self.v_s) ** self.alpha)) * (self.k_cs + (self.k_ms * tau_m - self.k_es * tau_e).abs())
        return self.k_v * w_abs + coulomb + static

    def reset(self, container: ModuleContainer) -> None:
        """Restart the friction model's input history of the resetting envs (rows env * num_tendons + tendon)."""
        if self.friction_model is None:
            return
        rows = container.reset_buf.unsqueeze(-1).expand_as(self.friction_model_scale).reshape(-1)
        # Before its first call the model has no per-row state yet (it allocates it fresh then).
        if self.friction_model.force_buffer.shape[0] == rows.numel() and bool(rows.any()):
            with torch.no_grad():
                self.friction_model.reset(rows)

    def _model_output(self, omega_m, theta_delta):
        """Learned model output (envelope or friction torque), one batched forward pass over every env and tendon."""
        with torch.no_grad():
            output = self.friction_model(omega_m.reshape(-1), theta_delta.reshape(-1)).view_as(omega_m)
        return output * (self.friction_model_gain * self.friction_model_scale)

    def step(self, container: ModuleContainer) -> None:
        spool = container.spool
        theta_m, omega_m, tau_e = spool.theta_m, spool.omega_m, spool.tau_e

        # The action is the desired spool angle, capped at 0: it never asks to unwind below the tared taut angle
        # (the hand controller applies the same cap on the robot).
        actions = spool.commanded_angle(container).clamp(min=0.0)
        tau_m = self._motor_torque(actions - theta_m)

        if not self.friction:
            tau_f_max = torch.zeros_like(tau_m)
        elif self.friction_model is not None:
            tau_f_max = self._model_output(omega_m, actions - theta_m)
        else:
            tau_f_max = self._friction_budget(tau_m, tau_e, omega_m)
        if self.friction and self.friction_model is not None and self.friction_model_output == "torque":
            tau_f = tau_f_max  # the model's friction torque itself
            tau_f_max = tau_f.abs()
        else:
            tau_f_max = tau_f_max.clamp(min=0.0)
            tau_f_stop = -(spool.j_eff / self.dt * omega_m + tau_m + tau_e)
            tau_f = torch.clamp(tau_f_stop, -tau_f_max, tau_f_max)

        tau = tau_m + tau_f
        # tau = tau_m
        if self.max_slider_velocity is not None:
            # Keep J_eff * (omega_next - omega_m) / dt = tau + tau_e within |omega_next| <= w_max.
            w_max = self.max_slider_velocity / spool.pully_radius
            inertia_dt = spool.j_eff / self.dt
            tau = torch.minimum(torch.maximum(tau, inertia_dt * (-w_max - omega_m) - tau_e), inertia_dt * (w_max - omega_m) - tau_e)
        force = tau / spool.pully_radius
        container.spool_motor_force_buf[:, spool.motor_indices] = force

        # Signals of the last step, shape (num_envs, num_model_tendons), for inspection tools.
        self.last_signals = {
            "theta_m": theta_m, "omega_m": omega_m, "tension": spool.tension, "tau_e": tau_e,
            "tau_m": tau_m, "tau_f": tau_f, "tau_f_max": tau_f_max, "force": force, "action": actions,
        }

        if self.debug:
            e, t = self.debug_env, self.debug_tendon
            print(" ".join(f"{name} {v[e, t].item(): .4f}" for name, v in self.last_signals.items()))

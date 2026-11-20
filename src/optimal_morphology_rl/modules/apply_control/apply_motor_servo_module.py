"""Servo law on the 'create_motor' joint servos: computes the joint motor torques."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import optimal_morphology_rl_assets
import torch

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


def _resolve_model_path(path: str) -> Path:
    """Absolute paths are used as is, relative ones resolve against the assets 'models/actuators' folder."""
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(optimal_morphology_rl_assets.__file__).resolve().parent / "models" / "actuators" / p


# M5 friction parameters (config key -> attribute), the same law and defaults as 'apply_m5_forces'.
M5_PARAMS = {
    "K_v": ("k_v", 0.0037641),
    "K_c": ("k_c", 0.00666241),
    "K_m": ("k_m", 0.447678),
    "K_e_wind": ("k_e_wind", 0.00129603),
    "K_e_unwind": ("k_e_unwind", 0.00129603),
    "v_s": ("v_s", 0.0628076),
    "alpha": ("alpha", 1.00722),
    "K_cs": ("k_cs", 0.000773996),
    "K_ms": ("k_ms", 0.303106),
    "K_es": ("k_es", 0.00117),
}
# Motor speed (rad/s) below which the friction direction is the one the motor is pushed in (tau_m + tau_e).
WIND_DIRECTION_EPS = 1e-3


@register_module("apply_motor_servo")
class ApplyMotorServoModule(BaseModule):
    """Position servo with friction on the joint motors of 'create_motor' (``container.motor``), every physics step.

    From the motor state (theta, omega) and the policy action, the change of the motor angle (rad):

        desired = max(theta + action, min_target)                      (held target with 'update_motor_target')
        i       = motor_position_gain * (desired - theta)              (current command, A)
        tau_m   = torque_constant * sign(i) * max(|i| - current_deadband, 0)
        tau_f   = clamp(-(J / dt * omega + tau_m), -tau_f_max, tau_f_max)   (Karnopp: stops the motor within a step,
                                                                              up to the friction budget)
        tau     = tau_m + tau_f, capped so |omega| stays within max_velocity on the next step

    The friction budget tau_f_max comes from a learned TorchScript envelope ``model(omega, desired - theta)``
    (``friction_model_path``, relative to the assets' models/actuators) or, without a model, from the M5 law with the
    constants ``M5_PARAMS`` (each a scalar or one value per motor; the defaults of 'apply_m5_forces'):

        tau_f_max = K_v |omega| + K_c + |K_m tau_m - K_e tau_e|
                    + exp(-(|omega| / v_s)^alpha) (K_cs + |K_ms tau_m - K_es tau_e|)

    with K_e = K_e_wind while the motor turns in +q (at standstill: the direction tau_m + tau_e pushes it), else
    K_e_unwind. The load torque tau_e is not measured on a joint servo (no tendon): it is 0, so the K_e terms only
    matter once a load estimate is added. Either budget is scaled by the training curriculum factor
    (``curriculum_factor``: 0 = no friction); ``friction: false`` leaves it out. J is the motor's ``j_eff`` (the joint
    armature). The torques go to ``container.motor_servo_force_buf`` (per robot motor), applied by
    'apply_motor_forces'; ``last_signals`` keeps tau_m, tau_f, tau_f_max, tau_inertia (= J / dt * omega) for the
    'motor_torques' observation.

    Config shape::

        apply_motor_servo:
          friction: true
          friction_model_path: 2026_11_13_mlp_dynamic_envelope_weak_gain.pt # or none: the M5 law below
          K_v: 0.0015 # M5 constants (used without friction_model_path), see M5_PARAMS
          ...
          motor_position_gain: 0.1 # A/rad
          torque_constant: 0.5 # Nm/A
          current_deadband: 0.05 # A
          max_velocity: 7.6 # rad/s (optional)
          min_target: -1.0 # rad (optional)
    """

    physics_rate = True  # the stop torque and the speed limit integrate over one physics step

    def finalize(self, container: ModuleContainer) -> None:
        if container.get("motor") is None:
            raise RuntimeError("apply_motor_servo requires 'create_motor' (container.motor) in init_modules.")

    def post_finalize(self, container: ModuleContainer) -> None:
        motor = container.motor
        device = container.device
        self.dt = motor.dt
        self.num_motors = motor.num_motors
        self.motor_position_gain = float(self.config.get("motor_position_gain", 0.1))
        self.torque_constant = float(self.config.get("torque_constant", 0.5))
        self.current_deadband = float(self.config.get("current_deadband", 0.05))
        if self.current_deadband < 0.0:
            raise RuntimeError("apply_motor_servo 'current_deadband' must be >= 0.")
        max_velocity = self.config.get("max_velocity")
        self.max_velocity = None if max_velocity is None else float(max_velocity)
        min_target = self.config.get("min_target")
        self.min_target = None if min_target is None else float(min_target)

        self.friction = bool(self.config.get("friction", True))
        self.friction_model = None
        self.friction_model_name = None
        for key, (attr, default) in M5_PARAMS.items():
            setattr(self, attr, self._per_motor(key, default, device))
        if torch.any(self.v_s <= 0.0):
            raise RuntimeError("apply_motor_servo 'v_s' entries must be positive.")
        model_path = self.config.get("friction_model_path")
        if self.friction and model_path:
            model_path = _resolve_model_path(str(model_path))
            if not model_path.is_file():
                raise RuntimeError(f"apply_motor_servo: friction model '{model_path}' not found.")
            self.friction_model = torch.jit.load(str(model_path), map_location=device)
            self.friction_model.eval()
            self.friction_model_name = model_path.name
            frequency = float(getattr(self.friction_model, "frequency", 1.0 / self.dt))
            if abs(frequency * self.dt - 1.0) > 1e-6:
                print(f"apply_motor_servo: WARNING friction model runs at {frequency:g} Hz, the physics at {1.0 / self.dt:g} Hz")
            columns = list(getattr(self.friction_model, "output_columns", []) or [])
            self.envelope_column = None
            if len(columns) > 1:
                if "friction_envelope_Nm" not in columns:
                    raise RuntimeError(f"apply_motor_servo: friction model '{model_path.name}' outputs {columns}, no 'friction_envelope_Nm'.")
                self.envelope_column = columns.index("friction_envelope_Nm")
            print(f"apply_motor_servo: friction envelope from '{model_path.name}'")
        elif self.friction:
            def show(values: torch.Tensor):
                values = [round(x, 6) for x in values.tolist()]
                return values[0] if len(set(values)) == 1 else values
            print("apply_motor_servo: M5 friction " + ", ".join(f"{k} {show(getattr(self, a))}" for k, (a, _) in M5_PARAMS.items()))

        container.motor_servo_force_buf = torch.zeros(
            (container.total_num_envs, container.robot.num_motors), device=device, dtype=torch.float32
        )
        self.last_signals: dict[str, torch.Tensor] | None = None

    def export_info(self, container: ModuleContainer) -> dict[str, Any]:
        info = {
            "physics_dt": self.dt,
            "motor_position_gain": self.motor_position_gain,
            "torque_constant": self.torque_constant,
            "current_deadband": self.current_deadband,
        }
        if self.friction_model_name is not None:
            info["friction_model"] = self.friction_model_name
        elif self.friction:
            info["m5"] = {key: getattr(self, attr).cpu().tolist() for key, (attr, _) in M5_PARAMS.items()}
        return {"motor_servo": info}

    def reset(self, container: ModuleContainer) -> None:
        """Zero the reset envs' signals (no command, no motion) and restart the friction model's rows."""
        reset_buf = container.reset_buf
        if not bool(reset_buf.any()):
            return
        if self.last_signals is not None:
            for name in ("tau_m", "tau_f", "tau_f_max", "tau_inertia"):
                self.last_signals[name] = torch.where(reset_buf.unsqueeze(-1), 0.0, self.last_signals[name])
        if self.friction_model is not None:
            rows = reset_buf.unsqueeze(-1).expand(-1, self.num_motors).reshape(-1)
            # Before its first call the model has no per-row state yet (it allocates it fresh then).
            if self.friction_model.force_buffer.shape[0] == rows.numel():
                with torch.no_grad():
                    self.friction_model.reset(rows)

    def _per_motor(self, key: str, default: float, device: torch.device) -> torch.Tensor:
        """A scalar or per-motor config value as a ``(num_motors,)`` tensor."""
        value = self.config.get(key, default)
        if isinstance(value, (list, tuple)):
            if len(value) != self.num_motors:
                raise RuntimeError(f"apply_motor_servo '{key}' has {len(value)} values, expected one per motor ({self.num_motors}).")
            return torch.tensor([float(x) for x in value], device=device, dtype=torch.float32)
        return torch.full((self.num_motors,), float(value), device=device, dtype=torch.float32)

    def _m5_budget(self, tau_m: torch.Tensor, tau_e: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        """M5: directional load-dependent friction with Stribeck decay (as 'apply_m5_forces')."""
        w_abs = omega.abs()
        winding = torch.where(w_abs > WIND_DIRECTION_EPS, omega > 0.0, tau_m + tau_e > 0.0)
        k_e = torch.where(winding, self.k_e_wind, self.k_e_unwind)
        coulomb = self.k_c + (self.k_m * tau_m - k_e * tau_e).abs()
        static = torch.exp(-((w_abs / self.v_s) ** self.alpha)) * (self.k_cs + (self.k_ms * tau_m - self.k_es * tau_e).abs())
        return self.k_v * w_abs + coulomb + static

    def _friction_budget(self, omega: torch.Tensor, error: torch.Tensor, tau_m: torch.Tensor, tau_e: torch.Tensor) -> torch.Tensor:
        if not self.friction:
            return torch.zeros_like(omega)
        if self.friction_model is None:
            budget = self._m5_budget(tau_m, tau_e, omega)
        else:
            with torch.no_grad():
                output = self.friction_model(omega.reshape(-1), error.reshape(-1))
            if self.envelope_column is not None:
                output = output[:, self.envelope_column]
            budget = output.view_as(omega)
        return (budget * self.curriculum_factor).clamp(min=0.0)

    def step(self, container: ModuleContainer) -> None:
        motor = container.motor
        theta, omega = motor.theta, motor.omega
        desired = theta + motor.commanded_delta_angle(container)
        if self.min_target is not None:
            desired = desired.clamp(min=self.min_target)
        error = desired - theta

        current = self.motor_position_gain * error
        tau_m = self.torque_constant * torch.sign(current) * torch.clamp(current.abs() - self.current_deadband, min=0.0)
        tau_e = torch.zeros_like(tau_m)  # the joint's load torque is not measured
        tau_f_max = self._friction_budget(omega, error, tau_m, tau_e)
        tau_inertia = motor.j_eff / self.dt * omega
        tau_f = torch.clamp(-(tau_inertia + tau_m), -tau_f_max, tau_f_max)
        tau = tau_m + tau_f
        if self.max_velocity is not None:
            # J (omega_next - omega) / dt = tau (the motor's own torques) within |omega_next| <= max_velocity.
            inertia_dt = motor.j_eff / self.dt
            tau = torch.minimum(torch.maximum(tau, inertia_dt * (-self.max_velocity - omega)), inertia_dt * (self.max_velocity - omega))
        container.motor_servo_force_buf[:, motor.motor_indices] = tau

        self.last_signals = {
            "theta": theta, "omega": omega, "desired": desired,
            "tau_m": tau_m, "tau_f": tau_f, "tau_f_max": tau_f_max, "tau_inertia": tau_inertia, "tau": tau,
        }

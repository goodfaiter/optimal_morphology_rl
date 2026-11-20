"""Observation source for the joint servo state of 'create_motor' (motor angles and velocities)."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.observations.observation_base_module import ObservationBaseModule
from optimal_morphology_rl.modules.observations.observation_manager_module import register_observation


def _motor(env: Any):
    motor = env.module_manager.container.get("motor")
    if motor is None:
        raise RuntimeError("motor_* observations require the 'create_motor' module.")
    return motor


@register_observation("motor_state")
class MotorStateObservation(ObservationBaseModule):
    """Motor (joint) angles and velocities of the 'create_motor' servos: ``[theta (num_motors), omega (num_motors)]``
    (rad, rad/s; omega the finite difference, ``create_motor.joint_velocity``). ``velocity: false`` / ``angle: false``
    keep only the other half.

    Config shape (under ``observation.modules``)::

        motor_state: {velocity: true, angle: true}
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.velocity = bool(self.config.get("velocity", True))
        self.angle = bool(self.config.get("angle", True))
        if not (self.velocity or self.angle):
            raise ValueError("motor_state: 'angle' and 'velocity' cannot both be false.")

    def export_info(self, container: Any) -> dict[str, Any]:
        return {"motor_state": {"angle": self.angle, "velocity": self.velocity}}

    def get_obs_dim(self, env: Any) -> int:
        return (int(self.velocity) + int(self.angle)) * _motor(env).num_motors

    def compute_observation(self, env: Any, out: torch.Tensor) -> None:
        motor = _motor(env)
        container = env.module_manager.container
        parts = []
        if self.angle:
            parts.append(container.robot.get_joint_pos_buf[:, motor.dof_indices])
        if self.velocity:
            parts.append(motor.joint_velocity(container))
        out[:] = torch.cat(parts, dim=-1)


@register_observation("motor_action")
class MotorActionObservation(ObservationBaseModule):
    """The policy's commanded change of the motor angle per 'create_motor' servo of the step just taken (rad,
    ``commanded_delta_angle``; 0 after a reset): ``[action (num_motors)]``.

    Config shape (under ``observation.modules``)::

        motor_action: {}
    """

    def get_obs_dim(self, env: Any) -> int:
        return _motor(env).num_motors

    def compute_observation(self, env: Any, out: torch.Tensor) -> None:
        out[:] = _motor(env).commanded_delta_angle(env.module_manager.container)


# The robot can compute tau_f_max (the friction model), tau_m (the servo law) and tau_inertia (J / dt * omega); tau_f
# needs the load, which only the sim knows.
MOTOR_TORQUE_SIGNALS = ("tau_f_max", "tau_f", "tau_m", "tau_inertia")


@register_observation("motor_torques")
class MotorTorquesObservation(ObservationBaseModule):
    """Servo torques of the current physics step per 'create_motor' servo, from ``apply_motor_servo.last_signals``
    (Nm): ``[signal 1 (num_motors), signal 2, ...]`` in the order of ``signals`` (default ``[tau_f_max]``).

    Config shape::

        motor_torques:
          signals: [tau_f_max]  # any of tau_f_max, tau_f, tau_m, tau_inertia
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.signals = list(self.config.get("signals", ["tau_f_max"]))
        unknown = [name for name in self.signals if name not in MOTOR_TORQUE_SIGNALS]
        if unknown or not self.signals:
            raise ValueError(f"motor_torques 'signals' must be a non-empty subset of {MOTOR_TORQUE_SIGNALS}, got {self.signals}.")

    def export_info(self, container: Any) -> dict[str, Any]:
        return {"motor_torques": {"signals": list(self.signals), "j_eff": container.motor.j_eff[0].cpu().tolist()}}

    def get_obs_dim(self, env: Any) -> int:
        return len(self.signals) * _motor(env).num_motors

    def compute_observation(self, env: Any, out: torch.Tensor) -> None:
        last = getattr(env.module_manager.get("apply_motor_servo"), "last_signals", None)
        if last is None:  # before the first control step (the initial reset)
            out.zero_()
            return
        out[:] = torch.cat([last[name] for name in self.signals], dim=-1)

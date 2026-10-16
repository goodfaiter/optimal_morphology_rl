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
    "motor_gain": "motor_gain",
    "K_v": "k_v",
    "K_c": "k_c",
    "K_m": "k_m",
    "K_e": "k_e",
    "v_s": "v_s",
    "alpha": "alpha",
    "K_cs": "k_cs",
    "K_ms": "k_ms",
    "K_es": "k_es",
}


def _resolve_model_path(path: str) -> Path:
    """Absolute paths are used as is, relative ones resolve against the assets 'models/actuators' folder."""
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(optimal_morphology_rl_assets.__file__).resolve().parent / "models" / "actuators" / p


@register_module("apply_m5_forces")
class ApplyM5ForcesModule(BaseModule):
    """Servo motor law with M5 friction on the spools of 'create_spool' (``container.spool``).

    From the spool state (theta_m, omega_m, tau_e) and the commanded spool angle:

        tau_m   = motor_gain * (action - theta_m)
        tau_f   = M5 friction, Karnopp stop torque capped by the friction budget
        F       = (tau_m + tau_f) / r                    (motor force on the slider DOF)

    The tendon pulls the slider with -T and the servo inertia is the slider's armature, so the
    simulator closes the loop  J_eff * domega_m = tau_m + tau_f + tau_e  with the spool's J_eff.

    With 'max_slider_velocity' (m/s) the servo speed is limited (vlearn does not enforce the slider
    joint's velocity limit): tau_m + tau_f is capped so that this loop keeps |omega_m| within
    max_slider_velocity / r on the next step.

    With 'friction_model_path' set, the M5 coefficients K_c, K_v, K_m, K_e, K_cs, K_ms, K_es are
    predicted each step by a TorchScript friction model (``model(omega, theta_desired, theta, tau_e)``,
    then ``model.last_coefficients``) instead of the configured constants, and v_s / alpha are the
    model's fixed Stribeck parameters. The budget itself is still computed here, on this module's
    tau_m, tau_e and omega_m, and the Karnopp stop torque still uses the simulator's J_eff. The model
    is stateful (input history per row): it runs one forward pass over num_envs x num_model_tendons
    rows and is reset per row on environment resets.

    The parameters (``motor_gain`` and the M5 coefficients, see ``M5_PARAMS``) are per env and
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

        self.motor_gain = coef("motor_gain", 0.105)
        self.k_v = coef("K_v", 0.0037641)
        self.k_c = coef("K_c", 0.00666241)
        self.k_m = coef("K_m", 0.447678)
        self.k_e = coef("K_e", 0.00129603)
        self.v_s = coef("v_s", 0.0628076)
        self.alpha = coef("alpha", 1.00722)
        self.k_cs = coef("K_cs", 0.000773996)
        self.k_ms = coef("K_ms", 0.303106)
        self.k_es = coef("K_es", 0.00117)
        if torch.any(self.v_s <= 0.0):
            raise RuntimeError("ApplyM5Forces config 'v_s' entries must be positive.")
        # Configured values per tendon; the attributes become per env so they can be randomized.
        num_envs = container.total_num_envs
        self.nominal = {key: getattr(self, attr).clone() for key, attr in M5_PARAMS.items()}
        for key, attr in M5_PARAMS.items():
            setattr(self, attr, self.nominal[key].expand(num_envs, num_tendons).clone())
        for key in ("pully_radius", "model_tendon_indices", "slider_joint_suffix", "velocity_source"):
            if key in self.config:
                raise RuntimeError(f"ApplyM5Forces: '{key}' belongs to the 'create_spool' (or 'update_robot') config.")

        # Optional learned friction budget (TorchScript, 200 Hz, inputs in rad, rad/s, Nm).
        self.friction_model = None
        model_path = self.config.get("friction_model_path")
        if model_path:
            model_path = _resolve_model_path(str(model_path))
            if not model_path.is_file():
                raise RuntimeError(f"ApplyM5Forces: friction model '{model_path}' not found.")
            self.friction_model = torch.jit.load(str(model_path), map_location=device)
            self.friction_model.eval()
            self.friction_model.reset()
            model_dt = float(getattr(self.friction_model, "dt", self.dt))
            if abs(model_dt - self.dt) > 1e-9:
                print(f"ApplyM5Forces: WARNING friction model dt {model_dt} s != env control dt {self.dt} s")
            model_gain = float(getattr(self.friction_model, "motor_current_gain", float("nan"))) * float(
                getattr(self.friction_model, "motor_position_gain", float("nan"))
            )
            names = list(self.friction_model.coefficient_names)
            expected = ["K_c", "K_v", "K_m", "K_e", "K_cs", "K_ms", "K_es"]
            if names != expected:
                raise RuntimeError(f"ApplyM5Forces: friction model coefficients {names} != {expected}.")
            self.model_v_s = float(self.friction_model.stribeck_velocity)
            self.model_alpha = float(self.friction_model.stribeck_exponent)
            print(f"ApplyM5Forces: friction budget from '{model_path.name}' (model motor gain {model_gain:.6g} Nm/rad)")

        container.spool_motor_force_buf = torch.zeros(
            (container.total_num_envs, container.robot.num_motors), device=device, dtype=torch.float32
        )

        # Optional debug print of one (env, tendon) pair. Off by default: .item() forces a GPU sync.
        max_v = self.config.get("max_slider_velocity")
        self.max_slider_velocity = None if max_v is None else float(max_v)
        if self.max_slider_velocity is not None and self.max_slider_velocity <= 0.0:
            raise RuntimeError("ApplyM5Forces config 'max_slider_velocity' must be positive (m/s).")

        self.debug = bool(self.config.get("debug", False))
        self.debug_env = int(self.config.get("debug_env", 0))
        self.debug_tendon = int(self.config.get("debug_tendon", 0))

    def _friction_budget(self, tau_m, tau_e, w_abs, coefficients=None):
        """M5: directional load-dependent friction with Stribeck decay (paper, eq. 9).

        coefficients: optional (..., 7) tensor of K_c, K_v, K_m, K_e, K_cs, K_ms, K_es predicted by the
        friction model (used with the model's v_s / alpha); defaults to the configured constants.
        """
        if coefficients is None:
            k_c, k_v, k_m, k_e = self.k_c, self.k_v, self.k_m, self.k_e
            k_cs, k_ms, k_es = self.k_cs, self.k_ms, self.k_es
            v_s, alpha = self.v_s, self.alpha
        else:
            k_c, k_v, k_m, k_e, k_cs, k_ms, k_es = coefficients.unbind(-1)
            v_s, alpha = self.model_v_s, self.model_alpha
        coulomb = k_c + (k_m * tau_m - k_e * tau_e).abs()
        static = torch.exp(-((w_abs / v_s) ** alpha)) * (k_cs + (k_ms * tau_m - k_es * tau_e).abs())
        return k_v * w_abs + coulomb + static

    def step(self, container: ModuleContainer) -> None:
        spool = container.spool
        theta_m, omega_m, tau_e = spool.theta_m, spool.omega_m, spool.tau_e

        actions = spool.commanded_angle(container)
        tau_m = self.motor_gain * (actions - theta_m)

        if self.friction_model is not None:
            with torch.no_grad():
                # Restart the input history of the reset rows (the model allocates and fills its
                # history itself on the first call, when its buffers do not match the row count yet).
                reset_rows = spool.reset_rows.unsqueeze(-1).expand_as(theta_m).reshape(-1)
                if self.friction_model.needs_init.numel() == reset_rows.numel() and bool(reset_rows.any()):
                    self.friction_model.reset(reset_rows)
                # Only the predicted M5 coefficients are used; the model's own tau_f is discarded.
                self.friction_model(omega_m.reshape(-1), actions.reshape(-1), theta_m.reshape(-1), tau_e.reshape(-1))
                self.last_coefficients = self.friction_model.last_coefficients.view(*theta_m.shape, -1)
            tau_f_max = self._friction_budget(tau_m, tau_e, omega_m.abs(), self.last_coefficients)
        else:
            tau_f_max = self._friction_budget(tau_m, tau_e, omega_m.abs())
        tau_f_stop = -(spool.j_eff / self.dt * omega_m + tau_m + tau_e)
        tau_f = torch.clamp(tau_f_stop, -tau_f_max, tau_f_max)

        tau = tau_m + tau_f
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

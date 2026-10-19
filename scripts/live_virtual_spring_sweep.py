"""Live sweep of a virtual M5 servo spool pulling a horizontal spring (no gravity, no simulator).

The same servo model as 'create_spool' + 'apply_m5_forces', but the load is a linear spring of stiffness
``k`` attached to the tendon instead of the finger, and the spool is integrated here:

    x       = r * theta_m                       (spring extension, tendon wound on the spool)
    T       = k * max(x, 0)                     (tendon tension; slack when unwound past zero)
    tau_e   = -r * T_measured                   (T_measured: previous-step tension, as in the sim)
    tau_m   = motor_gain * (action - theta_m)
    tau_f   = clamp(-(J/dt * omega_m + tau_m + tau_e), +-friction budget)   (M5 + Karnopp stop)
    F       = clamp((tau_m + tau_f) / r, motor force limits)
    omega_m += dt / J * (r * F - r * T),  theta_m += dt * omega_m  (semi-implicit Euler, travel-limited)

The servo matches the finger example (``plot_tendon_force_m5_sweep.py`` / the live finger
sweep): the adapt_finger ``test_env.yaml`` env is built once to read the 'create_spool' + 'apply_m5_forces'
module and the slider of the swept tendon, so the defaults (servo coefficients, motor gain,
pulley radius, J, dt = timestep * frame_skip, motor force limits, slider travel limits) are the
ones the simulator runs with. The slider is J; the matching slider armature J / r**2 (the mass
the simulator's slider must have) is shown live. The friction budget is the module's own
``_friction_budget``.

The action is ramped 0 -> sweep max -> 0 and the whole sweep is recomputed whenever a slider
moves; the previous sweep stays drawn in grey for comparison. "randomize K, v_s, alpha" draws
the M5 friction coefficients (K_v .. K_es, v_s, alpha) uniformly over their slider ranges.

Run with:  uv run python3 scripts/live_virtual_spring_sweep.py
"""

from __future__ import annotations

import contextlib
import io

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Button, CheckButtons, Slider

from optimal_morphology_rl.envs.modular_environment import ModularEnvironment
from optimal_morphology_rl.modules.apply_control.apply_m5_forces_module import ApplyM5ForcesModule
from plot_tendon_force_m5_sweep import TENDON, TENDON_CHOICES, build_config

# The sweep script selects the non-interactive Agg backend on import: switch to Tk after it.
plt.switch_backend("TkAgg")

SPRING_CONSTANT = 417.582417582  # N/m

# Slider: (attribute, label, min, max). The config keys of 'create_spool' (pully_radius) and 'apply_m5_forces' map to these attributes.
PARAM_SLIDERS = [
    ("spring_constant", "spring k (N/m)", 0.0, 1000.0),
    ("pully_radius", "pully radius (m)", 0.002, 0.03),
    ("motor_gain", "motor gain (Nm/rad)", 0.0, 0.5),
    ("inertia", "J (Nm s^2/rad)", 1e-5, 0.005),
    ("k_v", "K_v", 0.0, 0.02),
    ("k_c", "K_c", 0.0, 0.03),
    ("k_m", "K_m", 0.0, 1.5),
    ("k_e", "K_e", 0.0, 0.1),
    ("v_s", "v_s", 0.005, 0.3),
    ("alpha", "alpha", 0.2, 4.0),
    ("k_cs", "K_cs", 0.0, 0.05),
    ("k_ms", "K_ms", 0.0, 1.0),
    ("k_es", "K_es", 0.0, 0.1),
]
# Friction coefficients the "randomize" button draws, uniformly over their slider range.
RANDOMIZE_PARAMS = ("k_v", "k_c", "k_m", "k_e", "v_s", "alpha", "k_cs", "k_ms", "k_es")
# Servo coefficients read from the finger example's 'create_spool' + 'apply_m5_forces' modules.
MODULE_PARAMS = ("pully_radius", "motor_gain", "k_v", "k_c", "k_m", "k_e", "v_s", "alpha", "k_cs", "k_ms", "k_es")

# Plot rows: (y label, [(signal key, legend label), ...]).
PLOT_ROWS = [
    ("spool angle (rad)", [("action", "commanded"), ("theta_m", r"$\theta_m$")]),
    ("spring extension (mm)", [("extension_mm", "x")]),
    (r"spool velocity $\omega_m$ (rad/s)", [("omega_m", r"$\omega_m$")]),
    ("tendon tension (N)", [("tension", "T")]),
    (r"torques (N$\cdot$m)", [("tau_m", r"$\tau_m$"), ("tau_e", r"$\tau_e$"), ("tau_f", r"$\tau_f$"), ("tau_f_max", r"$\tau_{f,max}$")]),
    ("slider force (N)", [("force_command", "commanded"), ("force", "applied (capped)")]),
]
SIGNAL_KEYS = [key for _, series in PLOT_ROWS for key, _ in series]


class VirtualSpringRig:
    """Virtual servo spool + spring. Holds the coefficients under the module's attribute names
    (as 0-d tensors) so the module's ``_friction_budget`` runs on it unchanged.

    Reproduces what the simulator does around 'create_spool' + 'apply_m5_forces' on the finger example:

    - the module sees the tension *measured on the previous step* (``tension_lag``), while
      the tendon pulls the spool with the current tension,
    - the slider motor force is capped to the motor limits ``[force_low, force_high]`` (N),
    - the slider travel is limited to ``[travel_low, travel_high]`` (m), i.e. the spool angle
      to travel / r; at a limit the spool stops (inelastic).
    """

    def __init__(self, params: dict[str, float], dt: float) -> None:
        self.dt = dt
        self.tension_lag = True
        self.set_params(params)

    def set_params(self, params: dict[str, float]) -> None:
        for name, value in params.items():
            setattr(self, name, torch.tensor(float(value), dtype=torch.float64))

    def run(self, actions: np.ndarray, load_tension: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Sweep ``actions``. ``load_tension[i]`` (optional) replays the tendon tension acting
        during step i instead of the spring (e.g. recorded on the finger)."""
        r, k, dt, inertia = self.pully_radius, self.spring_constant, self.dt, self.inertia
        theta_low, theta_high = self.travel_low / r, self.travel_high / r
        theta_m = torch.zeros((), dtype=torch.float64)
        omega_m = torch.zeros((), dtype=torch.float64)
        measured_tension = torch.zeros((), dtype=torch.float64)  # previous-step tension
        trace = {key: np.zeros(len(actions)) for key in SIGNAL_KEYS}
        for i, action in enumerate(actions):
            if load_tension is None:
                tension = k * torch.clamp(r * theta_m, min=0.0)
            else:
                tension = torch.tensor(float(load_tension[i]), dtype=torch.float64)
            control_tension = measured_tension if self.tension_lag else tension

            # Servo law of 'apply_m5_forces' (on the measured tension).
            tau_e_control = -r * control_tension
            tau_m = self.motor_gain * (float(action) - theta_m)
            tau_f_max = ApplyM5ForcesModule._friction_budget(self, tau_m, tau_e_control, omega_m.abs())
            tau_f_stop = -(inertia / dt * omega_m + tau_m + tau_e_control)
            tau_f = torch.clamp(tau_f_stop, -tau_f_max, tau_f_max)
            force_command = (tau_m + tau_f) / r
            force = torch.clamp(force_command, self.force_low, self.force_high)

            values = {
                "action": action, "theta_m": theta_m, "extension_mm": 1000.0 * r * theta_m, "omega_m": omega_m,
                "tension": tension, "tau_m": tau_m, "tau_e": tau_e_control, "tau_f": tau_f, "tau_f_max": tau_f_max,
                "force_command": force_command, "force": force,
            }
            for key, value in values.items():
                trace[key][i] = float(value)

            # Plant: applied (capped) motor force and the current tension, travel-limited.
            omega_m = omega_m + dt / inertia * (r * force - r * tension)
            theta_m = theta_m + dt * omega_m
            if theta_m > theta_high or theta_m < theta_low:
                theta_m = torch.clamp(theta_m, theta_low, theta_high)
                omega_m = torch.zeros_like(omega_m)
            measured_tension = tension
        return trace


class LiveSpringSweep:
    def __init__(self) -> None:
        # Build the finger example once to read the servo the simulator actually runs.
        with contextlib.redirect_stdout(io.StringIO()):
            env = ModularEnvironment(build_config())
        m5 = env.module_manager.get("apply_m5_forces")
        spool = env.module_manager.container.spool
        art_def = env.module_manager.container.robot.art_def
        t = spool.model_tendon_indices.index(TENDON_CHOICES[TENDON])
        motor_def = art_def.get_motor_def(spool.motor_indices_list[t])
        dof_def = art_def.get_joint_dof_def(spool.dof_indices_list[t])
        # Per-tendon or per-env (env 0) tensors; the spool owns the radius, the force law the rest.
        self.defaults = {
            attr: float(getattr(spool if attr == "pully_radius" else m5, attr)[..., t].reshape(-1)[0].item()) for attr in MODULE_PARAMS
        }
        self.defaults["inertia"] = float(spool.j_eff[0, t].item())  # J the simulator integrates
        self.defaults["spring_constant"] = SPRING_CONSTANT
        # Fixed by the simulator's slider: motor force limits (N) and travel limits (m).
        self.limits = {
            "force_low": float(motor_def.low_limit), "force_high": float(motor_def.high_limit),
            "travel_low": float(dof_def.low_limit), "travel_high": float(dof_def.high_limit),
        }
        self.tendon_label = f"{TENDON.upper()} servo"
        dt = m5.dt
        del env, m5, spool, art_def

        self.params = dict(self.defaults)
        self.rig = VirtualSpringRig({**self.params, **self.limits}, dt)

        self.sweep_max = 2.0 * np.pi
        self.ramp_steps = 500
        self.rng = np.random.default_rng()
        self.batch_update = False  # set while several sliders move at once: recompute once at the end
        self._build_figure()
        self._recompute()

    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(15, 10))
        self.fig.canvas.manager.set_window_title("virtual M5 servo + spring")
        plot_grid = self.fig.add_gridspec(len(PLOT_ROWS), 1, left=0.06, right=0.62, top=0.95, bottom=0.06, hspace=0.35)
        self.axes, self.lines, self.ghost_lines = [], {}, {}
        for row, (ylabel, series) in enumerate(PLOT_ROWS):
            ax = self.fig.add_subplot(plot_grid[row], sharex=self.axes[0] if self.axes else None)
            for i, (key, label) in enumerate(series):
                style = "--" if key in ("action", "tau_f_max") else "-"
                (self.ghost_lines[key],) = ax.plot([], [], color="0.75", linewidth=1.0, linestyle=style)
                (self.lines[key],) = ax.plot([], [], color=f"C{i}", linewidth=1.2, linestyle=style, label=label)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(True, alpha=0.3)
            if len(series) > 1:
                ax.legend(loc="upper left", fontsize=7, ncol=len(series))
            self.axes.append(ax)
        self.axes[-1].set_xlabel("time (s)  [ramp up | ramp down]")
        self.title = self.axes[0].set_title("", fontsize=10)
        self.turn_lines = []

        self.sliders = {}
        top, height = 0.95, 0.032
        for i, (attr, label, vmin, vmax) in enumerate(PARAM_SLIDERS):
            ax = self.fig.add_axes([0.75, top - i * height, 0.18, height * 0.6])
            slider = Slider(ax, label, vmin, max(vmax, 2.0 * self.defaults[attr]), valinit=self.defaults[attr], valfmt="%.5g")
            slider.label.set_fontsize(8)
            slider.on_changed(lambda value, attr=attr: self._set_param(attr, value))
            self.sliders[attr] = slider

        y = top - len(PARAM_SLIDERS) * height - 0.02
        self.sweep_max_slider = Slider(self.fig.add_axes([0.75, y, 0.18, height * 0.6]), "sweep max (rad)", 0.5, 4.0 * np.pi, valinit=self.sweep_max, valfmt="%.3g")
        self.sweep_max_slider.on_changed(lambda value: self._set_sweep(sweep_max=float(value)))
        y -= height
        self.ramp_slider = Slider(self.fig.add_axes([0.75, y, 0.18, height * 0.6]), "steps per ramp", 50, 2000, valinit=self.ramp_steps, valstep=50)
        self.ramp_slider.on_changed(lambda value: self._set_sweep(ramp_steps=int(value)))
        y -= 0.08
        self.lag_check = CheckButtons(self.fig.add_axes([0.66, y, 0.12, 0.05]), ["tension lag (sim)"], [True])
        self.lag_check.on_clicked(self._toggle_lag)
        self.reset_button = Button(self.fig.add_axes([0.8, y + 0.005, 0.13, 0.04]), "reset parameters")
        self.reset_button.on_clicked(self._reset_params)
        self.randomize_button = Button(self.fig.add_axes([0.8, y - 0.04, 0.13, 0.04]), "randomize K, v_s, alpha")
        self.randomize_button.on_clicked(self._randomize_params)
        y -= 0.09
        self.info = self.fig.text(0.66, y, "", fontsize=9, va="top", family="monospace")

    def _set_param(self, attr: str, value: float) -> None:
        self.params[attr] = float(value)
        if not self.batch_update:
            self._recompute()

    def _set_sweep(self, sweep_max: float | None = None, ramp_steps: int | None = None) -> None:
        if sweep_max is not None:
            self.sweep_max = sweep_max
        if ramp_steps is not None:
            self.ramp_steps = ramp_steps
        self._recompute()

    def _toggle_lag(self, _label) -> None:
        self.rig.tension_lag = self.lag_check.get_status()[0]
        self._recompute()

    def _reset_params(self, _event) -> None:
        self.batch_update = True
        try:
            for slider in self.sliders.values():
                slider.reset()
        finally:
            self.batch_update = False
        self._recompute()

    def _randomize_params(self, _event) -> None:
        self.batch_update = True
        try:
            for attr in RANDOMIZE_PARAMS:
                slider = self.sliders[attr]
                slider.set_val(self.rng.uniform(slider.valmin, slider.valmax))
        finally:
            self.batch_update = False
        self._recompute()

    def _recompute(self) -> None:
        for key in SIGNAL_KEYS:
            self.ghost_lines[key].set_data(*self.lines[key].get_data())

        ramp_up = np.linspace(0.0, self.sweep_max, self.ramp_steps)
        actions = np.concatenate([ramp_up, ramp_up[::-1]])
        time = np.arange(len(actions)) * self.rig.dt
        self.rig.set_params({**self.params, **self.limits})
        trace = self.rig.run(actions)

        for key in SIGNAL_KEYS:
            self.lines[key].set_data(time, trace[key])
        for line in self.turn_lines:
            line.remove()
        self.turn_lines = [ax.axvline(time[self.ramp_steps], color="0.5", linestyle=":", linewidth=1.0) for ax in self.axes]
        self.axes[0].set_xlim(0.0, time[-1])
        for ax in self.axes:
            ax.relim()
            ax.autoscale_view(scalex=False)
        self.title.set_text(
            f"virtual {self.tendon_label} + spring k = {self.params['spring_constant']:.1f} N/m: 0 -> {self.sweep_max:.2f} -> 0 rad, "
            f"peak tension {trace['tension'].max():.1f} N"
        )
        r = self.params["pully_radius"]
        self.info.set_text(
            f"slider armature J/r^2 : {self.params['inertia'] / r**2:8.2f} kg\n"
            f"motor force limits    : [{self.limits['force_low']:.0f}, {self.limits['force_high']:.0f}] N\n"
            f"spool travel limits   : [{self.limits['travel_low'] / r:.2f}, {self.limits['travel_high'] / r:.2f}] rad"
        )
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    LiveSpringSweep()
    plt.show()

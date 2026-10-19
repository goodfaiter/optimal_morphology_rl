"""Live prismatic-force-m5 sweep with sliders for the servo / M5 friction parameters.

Runs the same headless adapt_finger sweep as ``plot_tendon_force_m5_sweep.py`` (raw
action ramped 0 -> sweep max -> 0, identity action scaling). Moving a slider writes the
new value straight into the 'create_spool' + 'apply_m5_forces' modules (all model tendons) and reruns
the sweep from a reset env; the previous complete sweep stays drawn in grey for
comparison. The sim is headless, so the whole trajectory runs in one go with the
signals recorded on the device and the plot is redrawn once at the end (a slider move
mid-sweep aborts it and restarts).

Run with:  uv run python3 scripts/live_prismatic_force_m5_sweep.py
Needs a display (TkAgg backend).
"""

from __future__ import annotations

import contextlib
import io
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Button, RadioButtons, Slider

from optimal_morphology_rl.envs.modular_environment import ModularEnvironment
from plot_tendon_force_m5_sweep import FINGER_JOINT, SETTLE_STEPS, TENDON_CHOICES, build_config

# The sweep script selects the non-interactive Agg backend on import: switch to Tk after it.
plt.switch_backend("TkAgg")
# plt.pause() re-shows (de-iconifies and raises) the window on every call while this is True, so the
# window could not be minimized: never raise it, and redraw with draw_idle + flush_events instead.
plt.rcParams["figure.raise_window"] = False

EVENT_CHECK_EVERY = 100  # sim steps between GUI event checks (a slider move aborts the sweep)

# Slider: (module attribute, label, min, max). Defaults come from the env config.
PARAM_SLIDERS = [
    ("pully_radius", "pully radius (m)", 0.002, 0.03),
    ("motor_gain", "motor gain (Nm/rad)", 0.0, 0.5),
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
# Antagonistic spring on the revolute finger joints ('antagonistic_spring' module), Nm/rad.
SPRING_SLIDER = ("spring_constant", "antag. spring (Nm/rad)", 0.0, 0.1)

# Plot rows: (title / y label, [(signal key, legend label), ...]).
PLOT_ROWS = [
    (r"spool angle (rad)", [("action", "commanded"), ("theta_m", r"$\theta_m$")]),
    ("finger angle (rad)", [("finger_angle", "finger joint")]),
    (r"spool velocity $\omega_m$ (rad/s)", [("omega_m", r"$\omega_m$")]),
    ("tendon tension (N)", [("tension", "T")]),
    (r"torques (N$\cdot$m)", [("tau_m", r"$\tau_m$"), ("tau_e", r"$\tau_e$"), ("tau_f", r"$\tau_f$"), ("tau_f_max", r"$\tau_{f,max}$")]),
    ("slider force (N)", [("force", "F")]),
]
SIGNAL_KEYS = [key for _, series in PLOT_ROWS for key, _ in series]


class LiveSweep:
    def __init__(self) -> None:
        config = build_config()
        # The module's per-step debug print forces a GPU sync for every signal.
        config["apply_m5_forces"]["debug"] = False
        with contextlib.redirect_stdout(io.StringIO()):
            self.env = ModularEnvironment(config)
        self.container = self.env.module_manager.container
        self.robot = self.container.robot
        self.m5 = self.env.module_manager.get("apply_m5_forces")
        self.spool = self.container.spool
        self.dof_names = self.robot.art_def.get_joint_dof_def_names()
        # Per-tendon or per-env (env 0) tensors, tendon 0.
        self.defaults = {attr: float(getattr(self._owner(attr), attr)[..., 0].reshape(-1)[0].item()) for attr, *_ in PARAM_SLIDERS}
        self.spring = self.env.module_manager.get("antagonistic_spring")
        self.defaults[SPRING_SLIDER[0]] = float(self.spring.spring_constants_buf[0, self.spring.revolute_motor_mask].max().item())
        self.actions = torch.zeros((self.env.total_num_envs,) + tuple(self.env.action_space.shape), device=self.env.device)
        # The freshly built sim state differs slightly from a post-reset one: reset and step once so
        # the first sweep starts like every later one (same parameters -> same trace).
        with contextlib.redirect_stdout(io.StringIO()):
            self.env.reset()
            self.env.step(self.actions)

        self.tendon = "pip"
        self.sweep_max = 1.0 * np.pi
        self.ramp_steps = 1200 # 12 [s]
        self.restart_requested = True
        self.last_sweep = None  # (time, trace) of the previous completed sweep, drawn in grey
        self._build_figure()

    # ---------------------------------------------------------------- figure
    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(15, 11))
        self.fig.canvas.manager.set_window_title("apply_m5_forces live sweep")
        plot_grid = self.fig.add_gridspec(len(PLOT_ROWS), 1, left=0.06, right=0.62, top=0.96, bottom=0.05, hspace=0.35)
        self.axes = []
        self.lines = {}
        self.ghost_lines = {}
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

        # Parameter sliders.
        self.sliders = {}
        top, height = 0.95, 0.032
        for i, (attr, label, vmin, vmax) in enumerate(PARAM_SLIDERS + [SPRING_SLIDER]):
            ax = self.fig.add_axes([0.75, top - i * height, 0.18, height * 0.6])
            slider = Slider(ax, label, vmin, max(vmax, 2.0 * self.defaults[attr]), valinit=self.defaults[attr], valfmt="%.5g")
            slider.label.set_fontsize(8)
            slider.on_changed(lambda value, attr=attr: self._set_param(attr, value))
            self.sliders[attr] = slider

        # Sweep controls.
        y = top - (len(PARAM_SLIDERS) + 1) * height - 0.02
        self.sweep_max_slider = Slider(self.fig.add_axes([0.75, y, 0.18, height * 0.6]), "sweep max (rad)", 0.5, 4.0 * np.pi, valinit=self.sweep_max, valfmt="%.3g")
        self.sweep_max_slider.on_changed(self._set_sweep_max)
        y -= height
        self.ramp_slider = Slider(self.fig.add_axes([0.75, y, 0.18, height * 0.6]), "steps per ramp", 50, 2000, valinit=self.ramp_steps, valstep=50)
        self.ramp_slider.on_changed(self._set_ramp_steps)

        y -= 0.1
        self.tendon_radio = RadioButtons(self.fig.add_axes([0.66, y, 0.1, 0.07]), list(TENDON_CHOICES), active=0)
        self.tendon_radio.on_clicked(self._set_tendon)
        self.reset_button = Button(self.fig.add_axes([0.8, y + 0.02, 0.13, 0.04]), "reset parameters")
        self.reset_button.on_clicked(self._reset_params)

    def _owner(self, attr: str):
        """The spool owns the radius, the force law owns the gains and friction coefficients."""
        return self.spool if attr == "pully_radius" else self.m5

    # ------------------------------------------------------------- callbacks
    def _set_param(self, attr: str, value: float) -> None:
        if attr == SPRING_SLIDER[0]:
            self._set_spring_constant(value)
            return
        getattr(self._owner(attr), attr)[:] = value
        if attr == "pully_radius":
            # Rotary inertia seen by the stop torque scales with r^2.
            self.spool.update_inertia()
        self.restart_requested = True

    def _set_spring_constant(self, value: float) -> None:
        # Spring constants are resampled from 'spring_constant_range' at every reset: pin the range too.
        if self.spring.spring_min is not None:
            self.spring.spring_min = self.spring.spring_max = float(value)
        if self.spring.spring_constants is not None:
            self.spring.spring_constants[:] = value
        self.spring.spring_constants_buf[:] = value
        self.restart_requested = True

    def _set_sweep_max(self, value: float) -> None:
        self.sweep_max = float(value)
        self.restart_requested = True

    def _set_ramp_steps(self, value: float) -> None:
        self.ramp_steps = int(value)
        self.restart_requested = True

    def _set_tendon(self, label: str) -> None:
        self.tendon = label
        for line in self.ghost_lines.values():
            line.set_data([], [])
        self.last_sweep = None
        self.restart_requested = True

    def _reset_params(self, _event) -> None:
        for slider in self.sliders.values():
            slider.reset()

    # ------------------------------------------------------------------ sweep
    def _start_sweep(self) -> None:
        self.restart_requested = False
        tendon_idx = TENDON_CHOICES[self.tendon]
        self.t = self.spool.model_tendon_indices.index(tendon_idx)
        self.finger_dof = self.dof_names.index(FINGER_JOINT[self.tendon])
        ramp_up = np.linspace(0.0, self.sweep_max, self.ramp_steps)
        self.sweep = np.concatenate([ramp_up, ramp_up[::-1]])
        # Whole sweep on the device: one action row per step, signals recorded without a host sync.
        self.sweep_actions = torch.tensor(self.sweep, device=self.env.device, dtype=torch.float32)
        self.time = np.arange(len(self.sweep)) * self.m5.dt
        self.trace_dev = torch.full((len(self.sweep), len(SIGNAL_KEYS)), float("nan"), device=self.env.device)
        self.step_idx = 0

        self.env.reset()
        self.actions[:] = 0.0
        for _ in range(SETTLE_STEPS):
            self.env.step(self.actions)

        self.axes[0].set_xlim(0.0, self.time[-1])
        for line in self.turn_lines:
            line.remove()
        self.turn_lines = [ax.axvline(self.time[self.ramp_steps], color="0.5", linestyle=":", linewidth=1.0) for ax in self.axes]
        self.title.set_text(f"apply_m5_forces sweep, {self.tendon.upper()} tendon {tendon_idx}: 0 -> {self.sweep_max:.2f} -> 0 rad")

    def _run_steps(self, num_steps: int) -> bool:
        """Advance the sweep; returns True when it completed."""
        end = min(self.step_idx + num_steps, len(self.sweep))
        joint_pos = self.robot.get_joint_pos_buf
        for i in range(self.step_idx, end):
            self.actions[:] = self.sweep_actions[i]
            self.env.step(self.actions)
            signals = self.m5.last_signals
            self.trace_dev[i] = torch.stack(
                [joint_pos[0, self.finger_dof] if key == "finger_angle" else signals[key][0, self.t] for key in SIGNAL_KEYS]
            )
        self.step_idx = end
        return end == len(self.sweep)

    def _fetch_trace(self) -> None:
        self.trace = self.trace_dev.T.cpu().numpy()

    def _redraw(self) -> None:
        for row, key in enumerate(SIGNAL_KEYS):
            self.lines[key].set_data(self.time, self.trace[row])
        for ax in self.axes:
            ax.relim()
            ax.autoscale_view(scalex=False)
        self.fig.canvas.draw_idle()

    def _show_ghost(self, time: np.ndarray, trace: np.ndarray) -> None:
        for row, key in enumerate(SIGNAL_KEYS):
            self.ghost_lines[key].set_data(time, trace[row])

    def run(self) -> None:
        plt.show(block=False)
        with contextlib.redirect_stdout(io.StringIO()):
            while plt.fignum_exists(self.fig.number):
                if not self.restart_requested:
                    # Idle until a parameter changes: the result stays on screen, the sim does not spin.
                    self.fig.canvas.flush_events()
                    time.sleep(0.02)
                    continue
                self._start_sweep()
                completed = False
                while not completed and not self.restart_requested and plt.fignum_exists(self.fig.number):
                    completed = self._run_steps(EVENT_CHECK_EVERY)
                    # Only pump GUI events, so a slider move aborts this sweep and restarts it.
                    self.fig.canvas.flush_events()
                if not completed:
                    continue
                self._fetch_trace()
                if self.last_sweep is not None:
                    self._show_ghost(*self.last_sweep)
                self.last_sweep = (self.time, self.trace)
                self._redraw()
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()


if __name__ == "__main__":
    LiveSweep().run()

"""Sweep the spool (create_spool + apply_m5_forces) action and plot the servo spool signals.

Drives the adapt_finger test environment (``test_env.yaml``, headless) with a
raw action ramped linearly from 0 up to ``SWEEP_MAX`` and back down to 0, and
records, for one
selected model tendon (set ``TENDON`` to "mcp" = tendon 0 or "pip" = tendon 1):

- the spool angle theta_m = q / r of the tendon's actuator slider vs. the
  commanded motor angle, and the matching finger joint angle,
- the spool velocity omega_m (slider position FD / r) and the tendon tension,
- the ``tau_m`` / ``tau_e`` / ``tau_f`` torque terms and the slider force
  ``(tau_m + tau_f) / r`` the 'apply_m5_forces' module applies.

The torques are computed right before each step from the same simulator
buffers and module state the module reads, using the module's own
coefficients and friction budget, and are checked against the slider force
the module actually wrote.

The action scaling is overridden to identity (``min_dof_scale: -1.0``,
``max_dof_scale: 1.0`` -> ``scale(x, -1, 1) == x``) so the raw action fed to
the env equals the commanded motor angle consumed by ``tau_m``.

Plots and the raw traces are written to /workspace/data/figs/.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from optimal_morphology_rl.envs.modular_environment import ModularEnvironment
from optimal_morphology_rl.utils.config import load_yaml_with_context

CONFIG_PATH = Path(
    "/workspace/optimal_morphology_rl/src/optimal_morphology_rl/envs/adapt_finger/test_env.yaml"
)
OUTPUT_DIR = Path("/workspace/data/figs")

SETTLE_STEPS = 50
SWEEP_STEPS = 1000  # per ramp (up, then down)
SWEEP_MAX = 2.0 * np.pi
TENDON = "mcp"  # "mcp" (tendon 0) or "pip" (tendon 1)
TENDON_CHOICES = {"mcp": 0, "pip": 1}
# Finger joint flexed by each tendon.
FINGER_JOINT = {"mcp": "Pinky_MCP", "pip": "Pinky_PIP"}


def build_config() -> dict:
    """Load test_env.yaml and apply the sweep overrides."""
    config = load_yaml_with_context(CONFIG_PATH)

    module_cfg = config["modules"]
    for phase in (
        "init_modules",
        "pre_physics_step_modules",
        "pre_gym_step_modules",
        "post_gym_step_modules",
        "post_physics_step_modules",
    ):
        if phase in module_cfg:
            module_cfg[phase] = [
                name
                for name in module_cfg[phase]
                if name not in ("render", "interactive_slider_control")
            ]

    env_cfg = config["create_rigid_vsim_envs"]
    env_cfg["rendering"] = False
    env_cfg["with_window"] = False

    pva = config.setdefault("process_variable_actions", {})
    pva["min_dof_scale"] = -1.0
    pva["max_dof_scale"] = 1.0

    return config


def main() -> None:
    tendon_idx = TENDON_CHOICES[TENDON]
    tendon_name = f"{TENDON.upper()} (tendon {tendon_idx})"

    config = build_config()
    env = ModularEnvironment(config)
    container = env.module_manager.container
    robot = container.robot
    m5 = env.module_manager.get("apply_m5_forces")
    spool = container.spool

    if tendon_idx not in spool.model_tendon_indices:
        raise RuntimeError(
            f"Tendon {tendon_idx} ({TENDON}) is not a model tendon of the config "
            f"(model_tendon_indices: {spool.model_tendon_indices})."
        )
    t = spool.model_tendon_indices.index(tendon_idx)
    slider_dof = spool.dof_indices_list[t]
    slider_motor = spool.motor_indices_list[t]
    finger_dof = robot.art_def.get_joint_dof_def_names().index(FINGER_JOINT[TENDON])
    action_slot = spool.action_slots_list[t]

    actions = torch.zeros(
        (env.total_num_envs,) + tuple(env.action_space.shape), device=env.device
    )

    ramp_up = np.linspace(0.0, SWEEP_MAX, SWEEP_STEPS)
    sweep = np.concatenate([ramp_up, ramp_up[::-1]])
    num_steps = len(sweep)
    keys = ("theta_m", "finger_angle", "omega_m", "tension", "tau_m", "tau_e", "tau_f", "force")
    traces = {key: np.zeros(num_steps) for key in keys}
    traces["commanded"] = np.zeros(num_steps)
    max_force_err = 0.0

    def coef(tensor: torch.Tensor) -> float:
        return float(tensor[..., t].reshape(-1)[0].item())  # per-tendon or per-env (env 0)

    r = coef(spool.pully_radius)

    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(SETTLE_STEPS):
            env.step(actions)

        for i, raw_action in enumerate(sweep):
            actions[:] = raw_action

            # Same inputs the module reads in its step (just_reset is clear after settling).
            q = float(robot.get_joint_pos_buf[0, slider_dof].item())
            if robot.velocity_source == "fd":
                qd = float(robot.calculated_joint_velocity[0, slider_dof].item())
            else:
                qd = float(robot.get_joint_vel_buf[0, slider_dof].item())
            tension = max(float(robot.get_tendon_force_buf[0, tendon_idx].item()), 0.0)

            theta_m = q / r
            omega_m = qd / r
            tau_e = -r * tension
            tau_m = coef(m5.motor_gain) * (raw_action - theta_m)
            tau_f_max = float(
                m5._friction_budget(
                    torch.tensor(tau_m, device=env.device),
                    torch.tensor(tau_e, device=env.device),
                    torch.tensor(abs(omega_m), device=env.device),
                )[0, t].item()
            )
            tau_f_stop = -(coef(spool.j_eff) / m5.dt * omega_m + tau_m + tau_e)
            tau_f = float(np.clip(tau_f_stop, -tau_f_max, tau_f_max))
            force = (tau_m + tau_f) / r

            env.step(actions)

            applied = float(container.spool_motor_force_buf[0, slider_motor].item())
            max_force_err = max(max_force_err, abs(applied - force))

            traces["commanded"][i] = float(
                env.scaled_act_buf[0, container.active_dof_slice][action_slot].item()
            )
            traces["theta_m"][i] = theta_m
            traces["finger_angle"][i] = float(robot.get_joint_pos_buf[0, finger_dof].item())
            traces["omega_m"][i] = omega_m
            traces["tension"][i] = tension
            traces["tau_m"][i] = tau_m
            traces["tau_e"][i] = tau_e
            traces["tau_f"][i] = tau_f
            traces["force"][i] = applied

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    labels = [
        ("theta_m", r"Spool angle $\theta_m$ (rad)"),
        ("finger_angle", f"{FINGER_JOINT[TENDON]} angle (rad)"),
        ("omega_m", r"Spool velocity $\omega_m$ (rad/s)"),
        ("tension", "Tendon tension (N)"),
        ("tau_m", r"$\tau_m$ (N$\cdot$m)"),
        ("tau_e", r"$\tau_e$ (N$\cdot$m)"),
        ("tau_f", r"$\tau_f$ (N$\cdot$m)"),
        ("force", "Slider force (N)"),
    ]

    # Plotted against time: ramp up, then ramp down, with the turnaround marked.
    time = np.arange(num_steps) * m5.dt
    t_turn = time[SWEEP_STEPS]

    fig, axes = plt.subplots(
        len(labels), 1, sharex=True, figsize=(10, 16), constrained_layout=True
    )
    for ax, (key, ylabel) in zip(axes, labels):
        if key == "theta_m":
            ax.plot(
                time,
                traces["commanded"],
                label="Commanded angle",
                linewidth=1.2,
                linestyle="--",
                color="C1",
            )
            ax.plot(time, traces[key], label="Spool angle", linewidth=1.2, color="C0")
        else:
            ax.plot(time, traces[key], label=tendon_name, linewidth=1.2, color="C0")
        ax.axvline(t_turn, color="0.5", linestyle=":", linewidth=1.0)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[0].set_title(
        f"apply_m5_forces: raw action ramp 0 $\\rightarrow$ {SWEEP_MAX:.2f} $\\rightarrow$ 0 rad ({tendon_name})"
    )
    axes[0].legend(loc="upper left")
    axes[-1].set_xlabel("Time (s)  [ramp up | ramp down]")
    plot_path = OUTPUT_DIR / f"prismatic_force_m5_sweep_{TENDON}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    npz_path = OUTPUT_DIR / f"prismatic_force_m5_sweep_{TENDON}.npz"
    np.savez(npz_path, time=time, action=sweep, **traces)

    print(f"Saved plot to {plot_path}")
    print(f"Saved traces to {npz_path}")
    print(f"  max |recomputed - applied| slider force: {max_force_err:.3e} N")
    for key in keys:
        print(f"  {key}: [{np.min(traces[key]):.4f}, {np.max(traces[key]):.4f}]")


if __name__ == "__main__":
    main()

"""Live matplotlib plot of the spool state and command, in a separate window / process."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module
from optimal_morphology_rl.utils.live_plot import LivePlot


@register_module("spool_plot")
class SpoolPlotModule(BaseModule):
    """Scrolling plots of every 'create_spool' spool of env ``env`` (debug / interactive use).

    One row per spool: the commanded spool angle and theta_m (rad), and omega_m (rad/s). The plot
    runs in its own subprocess (``utils/live_plot.py``), so drawing never stalls the simulation and the
    window can be minimized or closed (closing it just stops the plot). The time axis is the sim time
    since the env was built (it keeps running across resets). List it in
    ``pre_physics_step_modules`` after 'create_spool' (it reads the state 'create_spool' just read and
    the command 'process_variable_actions' just scaled).

    Config shape::

        spool_plot:
          every: 2         # sample every N control steps (each sample copies from the GPU)
          window_s: 5.0    # seconds of history shown
          fps: 20          # redraw rate of the plot window
          env: 0           # which env to plot
          enabled: true
          backend: TkAgg   # optional, "Agg" with save_path to run without a display
          save_path: null  # optional PNG of the last frame, written when the plot ends
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.enabled = bool(self.config.get("enabled", True))
        self.every = max(1, int(self.config.get("every", 2)))
        self.window_s = float(self.config.get("window_s", 5.0))
        self.fps = float(self.config.get("fps", 20.0))
        self.env_index = int(self.config.get("env", 0))
        self.backend = str(self.config.get("backend", "TkAgg"))
        self.save_path = self.config.get("save_path")
        self.plot: LivePlot | None = None
        self.steps = 0

    def finalize(self, container: ModuleContainer) -> None:
        if container.get("spool") is None:
            raise RuntimeError("SpoolPlotModule requires the 'create_spool' module (container.spool).")

    def post_finalize(self, container: ModuleContainer) -> None:
        if not self.enabled:
            return
        spool, robot = container.spool, container.robot
        rows = [robot.art_def.get_spatial_tendon_def(i).name.removesuffix("_tendon") for i in spool.model_tendon_indices]
        self.dt = spool.dt
        self.plot = LivePlot(
            {
                "title": "spool state",
                "rows": rows,
                "columns": [
                    {"ylabel": "spool angle (rad)", "signals": [("cmd", "command"), ("theta", r"$\theta_m$")]},
                    {"ylabel": "spool velocity (rad/s)", "signals": [("omega", r"$\omega_m$")]},
                ],
                "window_s": self.window_s,
                "fps": self.fps,
                "backend": self.backend,
                "save_path": self.save_path,
            }
        )

    def step(self, container: ModuleContainer) -> None:
        if self.plot is None or not self.plot.alive:
            return
        self.steps += 1
        if self.steps % self.every:
            return
        spool, e = container.spool, self.env_index
        sample = torch.stack([spool.commanded_angle(container)[e], spool.theta_m[e], spool.omega_m[e]]).cpu().tolist()
        self.plot.send(self.steps * self.dt, {"cmd": sample[0], "theta": sample[1], "omega": sample[2]})

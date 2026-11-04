"""Live matplotlib plot of the target and measured fingertip force, in a separate window / process."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module
from optimal_morphology_rl.utils.live_plot import LivePlot


@register_module("force_plot")
class ForcePlotModule(BaseModule):
    """Scrolling plot of the 'force_goal' target and the measured fingertip force (N) of env ``env``.

    Like 'spool_plot' (debug / interactive use): the plot runs in its own subprocess (``utils/live_plot.py``), so
    drawing never stalls the simulation and the window can be minimized or closed (closing it just stops the plot).
    The time axis is the sim time of the plotted env's episode: when that env resets, the plot starts again at 0. List
    it in ``post_physics_step_modules`` after 'force_goal' (it reads ``container.force_goal`` and
    ``container.fingertip_force`` of the same step).

    Config shape::

        force_plot:
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

    def post_finalize(self, container: ModuleContainer) -> None:
        if not self.enabled:
            return
        manager = container.env.module_manager
        if "force_goal" not in manager:
            raise RuntimeError("ForcePlotModule requires the 'force_goal' module (container.force_goal / fingertip_force).")
        self.dt = float(container.timestep) * float(container.frame_skip)
        self.plot = LivePlot(
            {
                "title": "fingertip force",
                "rows": [manager.get("force_goal").sensor_link],
                "columns": [{"ylabel": "force (N)", "signals": [("goal", "target"), ("force", "measured")]}],
                "window_s": self.window_s,
                "fps": self.fps,
                "backend": self.backend,
                "save_path": self.save_path,
            }
        )

    def reset(self, container: ModuleContainer) -> None:
        """Restart the time axis when the plotted env resets (the plotter then clears its history)."""
        reset_buf = container.get("reset_buf")
        if self.plot is not None and reset_buf is not None and bool(reset_buf[self.env_index]):
            self.steps = 0

    def step(self, container: ModuleContainer) -> None:
        if self.plot is None or not self.plot.alive:
            return
        self.steps += 1
        if self.steps % self.every:
            return
        e = self.env_index
        values = torch.stack([container.force_goal[e], container.fingertip_force[e]]).cpu().tolist()  # one GPU -> CPU copy
        self.plot.send(self.steps * self.dt, {"goal": [values[0]], "force": [values[1]]})

"""Scrolling matplotlib plots in a separate process, fed through a pipe.

The plotter runs as ``python -m optimal_morphology_rl.utils.live_plot <layout json>`` (a plain subprocess:
unlike ``multiprocessing`` it never re-imports the parent's main script) and reads one JSON sample per
line on stdin. Only numpy / matplotlib are imported there, so it starts quickly.

The window is never raised by the plotter: it avoids ``plt.pause`` (which re-shows and raises the figure on
every call while ``rcParams["figure.raise_window"]`` is True) and redraws with ``draw_idle`` +
``flush_events``, so it can be minimized, moved or closed freely. Closing it only ends the plotter.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from typing import Any


def run_plotter(layout: dict[str, Any], stream=None) -> list[float]:
    """Plotter loop: draws ``layout`` and appends the samples read from ``stream`` (default stdin).

    layout:
        title: window title
        rows: row labels (e.g. one per spool)
        columns: list of {"ylabel": str, "signals": [[key, legend label], ...]}
        window_s: seconds of history shown
        fps: redraw rate
        backend: matplotlib backend (default "TkAgg"; e.g. "Agg" to run without a display)
        save_path: optional PNG the last frame is written to when the plotter ends
    Returns the sample times shown at the end (used by the tests).

    Samples (one JSON line each): ``[t, {key: [value per row]}]``; end of stream ends the plotter. A
    sample earlier than the previous one starts over (e.g. the sender's episode was reset): the history
    is cleared.
    """
    import matplotlib

    matplotlib.use(layout.get("backend", "TkAgg"))
    matplotlib.rcParams["figure.raise_window"] = False
    import matplotlib.pyplot as plt

    stream = sys.stdin if stream is None else stream
    rows, columns = layout["rows"], layout["columns"]
    window_s, frame_dt = float(layout.get("window_s", 5.0)), 1.0 / float(layout.get("fps", 20.0))

    # Read the pipe on a thread so the GUI loop never blocks on it.
    samples: queue.Queue = queue.Queue()

    def reader():
        for line in stream:
            samples.put(json.loads(line))
        samples.put(None)

    threading.Thread(target=reader, daemon=True).start()

    fig, axes = plt.subplots(len(rows), len(columns), sharex=True, squeeze=False, figsize=(5 * len(columns), 1.6 * len(rows) + 1))
    fig.canvas.manager.set_window_title(layout.get("title", "live plot"))
    times: deque = deque()
    history = {key: [deque() for _ in rows] for column in columns for key, _ in column["signals"]}
    lines = {}
    for r, row in enumerate(rows):
        for c, column in enumerate(columns):
            ax = axes[r][c]
            for key, label in column["signals"]:
                (lines[key, r],) = ax.plot([], [], label=label, linewidth=1.2)
            ax.grid(True, alpha=0.3)
            if r == 0:
                ax.set_title(column["ylabel"], fontsize=10)
                ax.legend(loc="upper left", fontsize=8)
            if c == 0:
                ax.set_ylabel(row, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("sim time (s)")
    fig.tight_layout()
    plt.show(block=False)

    running = True
    while running and plt.fignum_exists(fig.number):
        deadline = time.monotonic() + frame_dt
        while running:  # take everything that arrived during this frame
            try:
                item = samples.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                break
            if item is None:
                running = False
                break
            t, values = item
            if times and t < times[-1]:  # time went back: new episode, start over
                times.clear()
                for per_row in history.values():
                    for h in per_row:
                        h.clear()
            times.append(t)
            for key, per_row in values.items():
                for r, value in enumerate(per_row):
                    history[key][r].append(value)
        while times and times[0] < times[-1] - window_s:
            times.popleft()
            for per_row in history.values():
                for h in per_row:
                    h.popleft()
        if times:
            t_list = list(times)
            for (key, r), line in lines.items():
                line.set_data(t_list, list(history[key][r]))
            for ax in axes.flat:
                ax.relim()
                ax.autoscale_view()
                ax.set_xlim(max(t_list[0], t_list[-1] - window_s), max(t_list[-1], t_list[0] + 1e-3))
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
    if layout.get("save_path"):
        fig.savefig(layout["save_path"], dpi=80)
    plt.close(fig)
    return list(times)


class LivePlot:
    """Parent-side handle: starts the plotter subprocess and sends it samples without ever blocking."""

    def __init__(self, layout: dict[str, Any]):
        self.process = subprocess.Popen(
            [sys.executable, "-m", "optimal_morphology_rl.utils.live_plot", json.dumps(layout)],
            stdin=subprocess.PIPE,
        )
        os.set_blocking(self.process.stdin.fileno(), False)
        self._pending = b""

    @property
    def alive(self) -> bool:
        return self.process.poll() is None

    def send(self, t: float, values: dict[str, list[float]]) -> None:
        """Send one sample; dropped while the plotter is behind (pipe full) or after its window closed."""
        if not self.alive:
            return
        data = self._pending + (json.dumps([t, values]) + "\n").encode()
        try:
            written = os.write(self.process.stdin.fileno(), data)
        except BlockingIOError:
            written = 0
        except (BrokenPipeError, OSError):
            return
        # A line already partly in the pipe must be finished first (keeps the stream line-aligned);
        # the samples after it are dropped.
        rest = data[written:]
        started = written > 0 or bool(self._pending)
        self._pending = rest[: rest.find(b"\n") + 1] if started else b""

    def close(self, timeout: float = 10.0) -> None:
        """End the plotter (it writes save_path, if set) and wait for it."""
        if self.alive:
            try:
                os.set_blocking(self.process.stdin.fileno(), True)
                if self._pending:
                    self.process.stdin.write(self._pending)
                self.process.stdin.close()
                self.process.wait(timeout=timeout)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self.process.kill()


if __name__ == "__main__":
    run_plotter(json.loads(sys.argv[1]))

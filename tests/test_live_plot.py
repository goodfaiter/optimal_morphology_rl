"""Tests for the subprocess live plotter (run headless with the Agg backend)."""

from pathlib import Path

from optimal_morphology_rl.utils.live_plot import LivePlot

LAYOUT = {
    "title": "test",
    "rows": ["a", "b"],
    "columns": [{"ylabel": "angle", "signals": [["cmd", "command"], ["theta", "theta"]]}, {"ylabel": "velocity", "signals": [["omega", "omega"]]}],
    "window_s": 1.0,
    "fps": 50,
    "backend": "Agg",
}


def test_plotter_runs_as_subprocess_and_saves_the_last_frame(tmp_path: Path) -> None:
    out = tmp_path / "frame.png"
    plot = LivePlot({**LAYOUT, "save_path": str(out)})
    assert plot.alive
    for i in range(500):
        plot.send(i * 0.005, {"cmd": [i * 0.01, 0.0], "theta": [i * 0.009, 0.0], "omega": [1.0, -1.0]})
    plot.close()
    assert plot.process.returncode == 0
    assert out.is_file() and out.stat().st_size > 0


def test_send_is_a_noop_once_the_plotter_has_exited(tmp_path: Path) -> None:
    plot = LivePlot(LAYOUT)
    plot.close()
    assert not plot.alive
    plot.send(0.0, {"cmd": [0.0, 0.0], "theta": [0.0, 0.0], "omega": [0.0, 0.0]})  # must not raise or block


def test_time_going_back_clears_the_history() -> None:
    import io
    import json

    from optimal_morphology_rl.utils.live_plot import run_plotter

    values = {"cmd": [0.0, 0.0], "theta": [0.0, 0.0], "omega": [0.0, 0.0]}
    first = [[i * 0.01, values] for i in range(100)]  # 0 .. 0.99 s
    after_reset = [[i * 0.01, values] for i in range(20)]  # episode restarted at 0
    stream = io.StringIO("".join(json.dumps(sample) + "\n" for sample in first + after_reset))

    times = run_plotter({**LAYOUT, "window_s": 10.0}, stream=stream)

    assert times == [sample[0] for sample in after_reset]


def test_spool_plot_restarts_its_clock_when_the_plotted_env_resets() -> None:
    import torch

    from optimal_morphology_rl.modules.module_container import ModuleContainer
    from optimal_morphology_rl.modules.visualization.spool_plot_module import SpoolPlotModule

    module = SpoolPlotModule({"env": 1})
    module.plot = object()  # a running plot
    container = ModuleContainer()
    module.steps = 50
    container.reset_buf = torch.tensor([True, False, False])
    module.reset(container)
    assert module.steps == 50  # another env reset
    container.reset_buf = torch.tensor([False, True, False])
    module.reset(container)
    assert module.steps == 0

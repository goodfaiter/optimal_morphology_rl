"""Unit tests for rl_games' training status display."""

import io

from rl_games.common.status_display import StatusDisplay, compact, duration, progress_bar


class _Terminal(io.StringIO):
    def isatty(self):
        return True


def test_terminal_redraws_the_status_in_place_and_keeps_events_above() -> None:
    out = _Terminal()
    display = StatusDisplay(out)
    display.update(["bar 1", "rewards 1"])
    display.update(["bar 2", "rewards 2"])
    display.event("saved best")
    text = out.getvalue()
    # The second update first moves up over and clears the two drawn lines.
    assert text.count("\x1b[1A\x1b[2K") == 4  # before the second update and before the event
    assert text.endswith("saved best\nbar 2\nrewards 2\n")
    assert display.drawn == 2


def test_log_file_gets_one_plain_line_per_update() -> None:
    out = io.StringIO()
    display = StatusDisplay(out)
    display.update(["epoch 1", "reward 3.0"])
    display.event("CURRICULUM: level 2/5")
    assert out.getvalue() == "epoch 1 | reward 3.0\nCURRICULUM: level 2/5\n"
    assert "\x1b" not in out.getvalue()


def test_formatting_helpers() -> None:
    assert progress_bar(0.5, width=4) == "[██░░]"
    assert progress_bar(2.0, width=2) == "[██]"
    assert compact(156000) == "156k" and compact(1_570_000) == "1.6M" and compact(42) == "42"
    assert duration(192) == "3:12" and duration(3725) == "1:02:05"


def test_other_prints_land_above_the_status(monkeypatch) -> None:
    import sys
    from rl_games.common import status_display

    out = _Terminal()
    monkeypatch.setattr(sys, "stdout", out)
    display = status_display.StatusDisplay()  # live: routes sys.stdout through itself
    try:
        display.update(["bar", "rewards"])
        print("=> saving checkpoint")
        display.update(["bar 2", "rewards 2"])
    finally:
        sys.stdout = out
    text = out.getvalue()
    assert text.endswith("=> saving checkpoint\nbar\nrewards\n" + "\x1b[1A\x1b[2K" * 2 + "bar 2\nrewards 2\n")

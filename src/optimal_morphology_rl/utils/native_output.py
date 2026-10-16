"""Silence output printed by native (C/C++) code, which ``contextlib.redirect_stdout`` cannot catch."""

from __future__ import annotations

import ctypes
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager

_libc = ctypes.CDLL(None)


@contextmanager
def suppress_native_stdout() -> Iterator[None]:
    """Redirect file descriptor 1 to /dev/null for the duration of the block.

    Python and C stdio buffers are flushed on entry and exit so nothing printed before or after
    the block is lost or swallowed. stderr is left alone, and exceptions propagate as usual.
    """
    sys.stdout.flush()
    _libc.fflush(None)
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        sys.stdout.flush()
        _libc.fflush(None)
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)

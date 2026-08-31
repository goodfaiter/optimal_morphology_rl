"""Shared helpers for hand-environment modules."""

from __future__ import annotations

from typing import Any


def get_reward_object_name(env: Any) -> str:
    """Return the configured reward object name from the shared container."""
    return env.module_manager.container.get("reward_object_name", "")

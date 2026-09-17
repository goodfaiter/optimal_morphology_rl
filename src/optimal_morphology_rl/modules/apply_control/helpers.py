"""Shared apply-control helper: robot dependency validation."""

from __future__ import annotations

from optimal_morphology_rl.modules.module_container import ModuleContainer


def require_robot(container: ModuleContainer, name: str) -> None:
    """Ensure the robot is populated by the create_robot module."""
    if container.get("robot") is None:
        raise RuntimeError(
            f"{name} requires 'robot' in the shared container. "
            f"Ensure the 'create_robot' module is listed before '{name}'."
        )

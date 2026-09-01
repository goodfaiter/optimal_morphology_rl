"""Module that refreshes fingertip force sensors after the physics step."""

from __future__ import annotations

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("update_force_sensors")
class UpdateForceSensorsModule(BaseModule):
    """Refreshes force sensor buffers created during finalize."""

    def step(self, container: ModuleContainer) -> None:
        """Refresh force sensor buffers."""
        force_sensors = container.get("force_sensors")
        if force_sensors is None:
            return
        force_sensors.update(container.gym)

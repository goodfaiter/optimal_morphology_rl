"""Module that refreshes scene object state buffers after the physics step."""

from __future__ import annotations

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("update_objects")
class UpdateObjectsModule(BaseModule):
    """Refreshes object state buffers for objects created by ``create_objects``."""

    def step(self, container: ModuleContainer) -> None:
        """Refresh state buffers for every loaded object."""
        objects = container.get("objects")
        if objects is None:
            return
        for obj in objects.values():
            obj.refresh_buffers(container.gym)

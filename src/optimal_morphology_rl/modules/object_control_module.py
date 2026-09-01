"""Module that dispatches pre-physics control hooks to scene objects."""

from __future__ import annotations

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("object_control")
class ObjectControlModule(BaseModule):
    """Calls pre-physics step hooks on objects created by ``create_objects``."""

    def step(self, container: ModuleContainer) -> None:
        """Dispatch pre-physics hooks to every loaded object."""
        create_objects = container.get("create_objects")
        if create_objects is None or create_objects.generator is None:
            return
        create_objects.generator.pre_physics_step(container.gym)

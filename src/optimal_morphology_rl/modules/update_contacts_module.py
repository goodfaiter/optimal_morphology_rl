"""Module that updates object-hand contact metrics after the physics step."""

from __future__ import annotations

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("update_contacts")
class UpdateContactsModule(BaseModule):
    """Calls ``update`` on the Contacts helper created by ``create_contacts``."""

    def step(self, container: ModuleContainer) -> None:
        """Refresh contact metrics."""
        contacts = container.get("contacts")
        if contacts is None:
            return
        contacts.update()

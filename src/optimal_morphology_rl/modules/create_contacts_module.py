"""Module that creates the object-hand contact helper."""

from __future__ import annotations

from typing import Any

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.contacts import Contacts
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("create_contacts")
class CreateContactsModule(BaseModule):
    """Wraps the Contacts helper and exposes it on the shared container."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.link_names = self.config.get("link_names", ["distal"])

    def finalize(self, container: ModuleContainer) -> None:
        """Verify dependencies."""
        if container.get("robot") is None or container.get("reward_object") is None:
            raise RuntimeError(
                "CreateContactsModule requires 'robot' and 'reward_object' in the shared container."
            )

        if container.get("create_objects") is None:
            raise RuntimeError(
                "CreateContactsModule requires 'create_objects' in the shared container."
            )

    def post_finalize(self, container: ModuleContainer) -> None:
        """Create the Contacts helper now that env_group and object offsets exist."""
        reward_object_contact_link_name = self._get_reward_object_contact_link_name(
            container.reward_object_name
        )
        reward_object_link_offset = container.create_objects.get_object_link_offset(
            container.reward_object_name
        )

        self.contacts = Contacts(
            env=container,
            reward_object=container.reward_object,
            reward_object_link_name=reward_object_contact_link_name,
            link_names=self.link_names,
            reward_object_link_offset=reward_object_link_offset,
        )
        container.contacts = self.contacts

    @staticmethod
    def _get_reward_object_contact_link_name(reward_object_name: str) -> str:
        if reward_object_name in ("button", "button_difficult"):
            return "button"
        if reward_object_name == "drawer":
            return "handle"
        return ""

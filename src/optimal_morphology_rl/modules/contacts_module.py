"""Module that computes object-hand contact metrics."""

from __future__ import annotations

from typing import Any

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.contacts import Contacts
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("contacts")
class ContactsModule(BaseModule):
    """Wraps the Contacts helper and exposes it on the shared container."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.link_names = self.config.get("link_names", ["distal"])

    def post_finalize(self, env: Any) -> None:
        container = env.module_manager.container
        if container.get("robot") is None or container.get("reward_object") is None:
            raise RuntimeError(
                "ContactsModule requires 'robot' and 'reward_object' in the shared container."
            )

        reward_object_contact_link_name = self._get_reward_object_contact_link_name(
            container.reward_object_name
        )

        object_generator = container.get("object_generator")
        if object_generator is None:
            raise RuntimeError(
                "ContactsModule requires 'object_generator' in the shared container."
            )
        reward_object_link_offset = object_generator.get_object_link_offset(
            container.reward_object_name
        )

        self.contacts = Contacts(
            env=env,
            reward_object=container.reward_object,
            reward_object_link_name=reward_object_contact_link_name,
            link_names=self.link_names,
            reward_object_link_offset=reward_object_link_offset,
        )
        container.contacts = self.contacts

    def post_physics_step(self, env: Any) -> None:
        self.contacts.update()

    @staticmethod
    def _get_reward_object_contact_link_name(reward_object_name: str) -> str:
        if reward_object_name in ("button", "button_difficult"):
            return "button"
        if reward_object_name == "drawer":
            return "handle"
        return ""

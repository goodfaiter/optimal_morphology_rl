"""Module that creates object-hand contact buffers and lookup tables."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("create_contacts")
class CreateContactsModule(BaseModule):
    """Creates and exposes contact query buffers on the shared container."""

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
        """Allocate contact buffers and build lookup tables."""
        self.device = container.device
        self.gym = container.gym
        self.robot = container.robot
        self.reward_object = container.reward_object
        self.reward_object_name = container.reward_object_name
        self.create_objects = container.create_objects

        self.max_contact_pairs_per_env = container.max_contact_pairs_per_env
        self.total_num_envs = container.total_num_envs
        self.num_links = self.robot.num_links

        size = self.max_contact_pairs_per_env * self.total_num_envs

        # Contact query buffers populated by gym.get_rigid_contacts.
        self.contact_normals_buf = torch.zeros(
            (size, 3), dtype=torch.float32, device=self.device
        )
        self.contact_point_seps_buf = torch.zeros(
            (size, 4), dtype=torch.float32, device=self.device
        )
        self.contact_id_a_buf = torch.zeros(
            (size, 4), dtype=torch.uint32, device=self.device
        )
        self.contact_id_b_buf = torch.zeros(
            (size, 4), dtype=torch.uint32, device=self.device
        )

        # Cache transform/contact lookup tables.
        max_envs_in_set = max(container.num_envs)
        self.contact_env_lookup = torch.full(
            (len(container.num_envs), max_envs_in_set),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self.reward_object_transform_index_by_env = torch.full(
            (self.total_num_envs,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self.hand_transform_indices_by_env = torch.full(
            (self.total_num_envs, self.num_links),
            -1,
            dtype=torch.long,
            device=self.device,
        )

        self.hand_transform_indices_by_env[:, :] = torch.arange(
            self.num_links, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        reward_object_contact_link_name = self._get_reward_object_contact_link_name(
            self.reward_object_name
        )
        reward_object_link_offset = container.create_objects.get_object_link_offset(
            self.reward_object_name
        )
        self.reward_object_transform_index_by_env[:] = self._compute_reward_object_transform_index(
            reward_object_contact_link_name,
            reward_object_link_offset,
        )

        env_flat_index = 0
        for set_index, env_set in enumerate(container.env_sets):
            num_envs_in_set = env_set.get_num_environments()
            for env_index in range(num_envs_in_set):
                self.contact_env_lookup[set_index, env_index] = env_flat_index
                env_flat_index += 1

        # Link mask for monitored hand links.
        link_name_set = {name.lower() for name in self.link_names}
        self.monitored_link_mask = torch.zeros(
            self.num_links, dtype=torch.bool, device=self.device
        )
        for name in link_name_set:
            for i in range(self.num_links):
                link_def = self.robot.art_def.get_link_def(i)
                if link_def.name.lower().endswith(name):
                    self.monitored_link_mask[i] = True
        if not torch.any(self.monitored_link_mask):
            raise ValueError("No monitored hand links were found.")

        # Output buffers.
        self.object_hand_contact_buf = torch.zeros(
            (self.total_num_envs,), device=self.device, dtype=torch.float32
        )
        self.object_hand_contact_count_buf = torch.zeros(
            (self.total_num_envs,), device=self.device, dtype=torch.float32
        )

        # Mask of touched links per env to deduplicate contacts.
        self.env_link_touch = torch.zeros(
            (self.total_num_envs, self.num_links), dtype=torch.bool, device=self.device
        )

        container.contacts = self

    def _compute_reward_object_transform_index(
        self,
        reward_object_link_name: str,
        reward_object_link_offset: int | None = None,
    ) -> torch.Tensor:
        """Return the global transform-table index for the reward-object link.

        The global transform table is laid out as::

            [hand links][object 0 links][object 1 links]...

        Args:
            reward_object_link_offset: Pre-computed cumulative link offset for
                the reward object. If ``None``, it is read from the object
                generator for backward compatibility.
        """
        if reward_object_link_offset is None:
            reward_object_link_offset = self.create_objects.get_object_link_offset(
                self.reward_object_name
            )
        num_reward_object_links = self.reward_object.get_link_offset()
        start_offset = reward_object_link_offset - num_reward_object_links

        if hasattr(self.reward_object, "art_def") and self.reward_object.art_def is not None:
            art_def = self.reward_object.art_def
            link_index = None
            for i in range(art_def.get_num_link_defs()):
                if art_def.get_link_def(i).name == reward_object_link_name:
                    link_index = i
                    break
            if link_index is None:
                raise ValueError(
                    f"Reward object link '{reward_object_link_name}' not found in "
                    f"object '{self.reward_object.name}'. Available links: "
                    f"{[art_def.get_link_def(i).name for i in range(art_def.get_num_link_defs())]}"
                )
        else:
            # Rigid body: only one transform handle exists.
            link_index = 0

        return self.num_links + start_offset + link_index

    @staticmethod
    def _get_reward_object_contact_link_name(reward_object_name: str) -> str:
        if reward_object_name in ("button", "button_difficult"):
            return "button"
        if reward_object_name == "drawer":
            return "handle"
        return ""

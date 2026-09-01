"""Module that updates object-hand contact metrics after the physics step."""

from __future__ import annotations

import torch
import vlearn as v

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("update_contacts")
class UpdateContactsModule(BaseModule):
    """Queries rigid contacts and computes object-hand contact metrics."""

    def step(self, container: ModuleContainer) -> None:
        """Refresh contact metrics using buffers owned by ``create_contacts``."""
        contacts = container.get("contacts")
        if contacts is None:
            return

        contacts.object_hand_contact_buf.zero_()
        contacts.object_hand_contact_count_buf.zero_()
        contacts.env_link_touch.zero_()

        max_contacts = contacts.max_contact_pairs_per_env * contacts.total_num_envs
        num_contacts = container.gym.get_rigid_contacts(
            v.wrap_gpu_buffer(contacts.contact_normals_buf),
            v.wrap_gpu_buffer(contacts.contact_point_seps_buf),
            v.wrap_gpu_buffer(contacts.contact_id_a_buf),
            v.wrap_gpu_buffer(contacts.contact_id_b_buf),
            max_contacts,
        )

        num_stored = min(num_contacts, max_contacts)
        if num_stored <= 0:
            return

        id_a = contacts.contact_id_a_buf[:num_stored].to(torch.long)
        id_b = contacts.contact_id_b_buf[:num_stored].to(torch.long)

        env_a = contacts.contact_env_lookup[id_a[:, 1], id_a[:, 2]]
        env_b = contacts.contact_env_lookup[id_b[:, 1], id_b[:, 2]]
        same_env = env_a == env_b
        valid_env = torch.logical_and(env_a >= 0, env_b >= 0)
        valid_contact = torch.logical_and(same_env, valid_env)
        if not torch.any(valid_contact):
            return

        env_indices = env_a.clamp_min(0)

        object_indices = contacts.reward_object_transform_index_by_env[env_indices]
        a_is_object = id_a[:, 3] == object_indices
        b_is_object = id_b[:, 3] == object_indices

        hand_indices = contacts.hand_transform_indices_by_env[env_indices]
        monitored_hand_indices = hand_indices[:, contacts.monitored_link_mask]
        a_is_hand = torch.any(
            id_a[:, 3].unsqueeze(1) == monitored_hand_indices, dim=1
        )
        b_is_hand = torch.any(
            id_b[:, 3].unsqueeze(1) == monitored_hand_indices, dim=1
        )

        object_hand_contact = torch.logical_and(
            valid_contact,
            torch.logical_or(
                torch.logical_and(a_is_object, b_is_hand),
                torch.logical_and(b_is_object, a_is_hand),
            ),
        )

        if torch.any(object_hand_contact):
            contact_env_indices = env_indices[object_hand_contact]
            a_is_hand_contact = a_is_hand[object_hand_contact]
            hand_link_indices = torch.where(
                a_is_hand_contact,
                id_a[object_hand_contact, 3],
                id_b[object_hand_contact, 3],
            )

            # Mark touched links per environment to deduplicate multiple contact pairs.
            contacts.env_link_touch[contact_env_indices, hand_link_indices] = True

            # Which envs had any monitored-link touch.
            touched_envs = torch.nonzero(contacts.env_link_touch.any(dim=1)).squeeze(-1)
            if touched_envs.numel() > 0:
                contacts.object_hand_contact_buf[touched_envs] = 1.0

            # Per-env unique contact counts = number of touched links per env.
            contacts.object_hand_contact_count_buf[:] = contacts.env_link_touch.sum(
                dim=1
            ).to(torch.float32)

"""Module that updates object-hand contact metrics after the physics step."""

from __future__ import annotations

import torch
import vlearn as v

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@torch.jit.script
def _update_contacts_jit(
    contact_id_a_buf: torch.Tensor,
    contact_id_b_buf: torch.Tensor,
    contact_env_lookup: torch.Tensor,
    reward_object_transform_index_by_env: torch.Tensor,
    hand_transform_indices_by_env: torch.Tensor,
    monitored_link_mask: torch.Tensor,
    total_num_envs: int,
    num_links: int,
    num_stored: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Process raw rigid-contact data into object-hand contact metrics.

    Args:
        contact_id_a_buf: (M, 4) contact IDs for body A.
        contact_id_b_buf: (M, 4) contact IDs for body B.
        contact_env_lookup: (num_sets, max_envs_in_set) flat env index or -1.
        reward_object_transform_index_by_env: (N,) global transform index.
        hand_transform_indices_by_env: (N, num_links) hand link transform indices.
        monitored_link_mask: (num_links,) bool mask of links to monitor.
        total_num_envs: number of parallel environments.
        num_links: number of hand links.
        num_stored: number of contacts returned by the simulation.

    Returns:
        env_link_touch: (N, num_links) bool touch mask.
        object_hand_contact_buf: (N,) binary contact indicator.
        object_hand_contact_count_buf: (N,) number of touched monitored links.
    """
    env_link_touch = torch.zeros(
        (total_num_envs, num_links), dtype=torch.bool, device=contact_id_a_buf.device
    )
    object_hand_contact_buf = torch.zeros(
        total_num_envs, dtype=torch.float32, device=contact_id_a_buf.device
    )
    object_hand_contact_count_buf = torch.zeros(
        total_num_envs, dtype=torch.float32, device=contact_id_a_buf.device
    )

    if num_stored <= 0:
        return env_link_touch, object_hand_contact_buf, object_hand_contact_count_buf

    id_a = contact_id_a_buf[:num_stored].to(torch.int64)
    id_b = contact_id_b_buf[:num_stored].to(torch.int64)

    env_a = contact_env_lookup[id_a[:, 1], id_a[:, 2]]
    env_b = contact_env_lookup[id_b[:, 1], id_b[:, 2]]
    same_env = env_a == env_b
    valid_env = torch.logical_and(env_a >= 0, env_b >= 0)
    valid_contact = torch.logical_and(same_env, valid_env)

    if not torch.any(valid_contact):
        return env_link_touch, object_hand_contact_buf, object_hand_contact_count_buf

    env_indices = env_a.clamp_min(0)

    object_indices = reward_object_transform_index_by_env[env_indices]
    a_is_object = id_a[:, 3] == object_indices
    b_is_object = id_b[:, 3] == object_indices

    hand_indices = hand_transform_indices_by_env[env_indices]
    monitored_hand_indices = hand_indices[:, monitored_link_mask]
    a_is_hand = torch.any(id_a[:, 3].unsqueeze(1) == monitored_hand_indices, dim=1)
    b_is_hand = torch.any(id_b[:, 3].unsqueeze(1) == monitored_hand_indices, dim=1)

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
        env_link_touch[contact_env_indices, hand_link_indices] = True

    touched_envs = torch.nonzero(env_link_touch.any(dim=1)).squeeze(-1)
    if touched_envs.numel() > 0:
        object_hand_contact_buf[touched_envs] = 1.0

    object_hand_contact_count_buf = env_link_touch.sum(dim=1).to(torch.float32)

    return env_link_touch, object_hand_contact_buf, object_hand_contact_count_buf


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

        (
            contacts.env_link_touch,
            contacts.object_hand_contact_buf,
            contacts.object_hand_contact_count_buf,
        ) = _update_contacts_jit(
            contacts.contact_id_a_buf,
            contacts.contact_id_b_buf,
            contacts.contact_env_lookup,
            contacts.reward_object_transform_index_by_env,
            contacts.hand_transform_indices_by_env,
            contacts.monitored_link_mask,
            contacts.total_num_envs,
            contacts.num_links,
            num_stored,
        )

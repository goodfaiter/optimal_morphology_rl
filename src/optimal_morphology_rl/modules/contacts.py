from typing import List

import torch
import vlearn as v
from vlearn import gym

# TODO: Refactor to avoid this hack to import from the vlearn repo.
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "/workspace/vlearn/train/envs/"))
from environment import EnvironmentGpu


class Contacts:
    """Helper to compute object-hand contact metrics using environment buffers.

    This class keeps a reference to the environment instance and reads/writes
    the same GPU-backed tensors the env exposes. It also caches transform
    lookup tables that were previously computed on the environment.
    """

    def __init__(self, env: EnvironmentGpu, link_names: List[str] | None = None) -> None:
        self.env: EnvironmentGpu = env
        self.device = self.env.device
        self.gym: gym.Gym = self.env.gym

        # Metadata
        self.max_contact_pairs_per_env = self.env.max_contact_pairs_per_env
        self.total_num_envs = self.env.total_num_envs
        self.num_links = self.env.num_links

        # Contact query buffers (owned by helper)
        self.contact_normals_buf = torch.zeros(
            (self.max_contact_pairs_per_env * self.total_num_envs, 3), dtype=torch.float32, device=self.device
        )
        self.contact_point_seps_buf = torch.zeros(
            (self.max_contact_pairs_per_env * self.total_num_envs, 4), dtype=torch.float32, device=self.device
        )
        self.contact_id_a_buf = torch.zeros(
            (self.max_contact_pairs_per_env * self.total_num_envs, 4), dtype=torch.uint32, device=self.device
        )
        self.contact_id_b_buf = torch.zeros(
            (self.max_contact_pairs_per_env * self.total_num_envs, 4), dtype=torch.uint32, device=self.device
        )

        # Cache transform/contact lookup tables that were previously on env
        max_envs_in_set = max(self.env.num_envs)
        self.contact_env_lookup = torch.full((len(self.env.num_envs), max_envs_in_set), -1, dtype=torch.long, device=self.device)
        self.reward_object_transform_index_by_env = torch.full((self.total_num_envs,), -1, dtype=torch.long, device=self.device)
        self.table_transform_index_by_env = torch.full((self.total_num_envs,), -1, dtype=torch.long, device=self.device)
        self.hand_transform_indices_by_env = torch.full((self.total_num_envs, self.num_links), -1, dtype=torch.long, device=self.device)

        self.hand_transform_indices_by_env[:, :] = torch.arange(self.num_links, dtype=torch.long, device=self.device).unsqueeze(0)
        self.table_transform_index_by_env[:] = self.num_links
        reward_object_offset = self.env.object_creation_order.index(self.env.reward_object)
        self.reward_object_transform_index_by_env[:] = self.num_links + 1 + reward_object_offset

        env_flat_index = 0
        for set_index, env_set in enumerate(self.env.env_sets):
            num_envs_in_set = env_set.get_num_environments()
            for env_index in range(num_envs_in_set):
                self.contact_env_lookup[set_index, env_index] = env_flat_index
                env_flat_index += 1

        # Link mask owned by helper and supplied by the environment.
        if link_names is None:
            link_names = []

        link_name_set = {name.lower() for name in link_names}
        self.monitored_link_mask = torch.zeros(self.num_links, dtype=torch.bool, device=self.device)

        for name in link_name_set:
            for i in range(self.num_links):
                link_def = self.env.art_def.get_link_def(i)
                if name in link_def.name.lower():
                    self.monitored_link_mask[i] = True

        if not torch.any(self.monitored_link_mask):
            raise ValueError("No monitored hand links were found.")

        # Output buffers owned by helper
        self.object_hand_contact_buf = torch.zeros((self.total_num_envs,), device=self.device, dtype=torch.float32)
        self.object_hand_contact_count_buf = torch.zeros((self.total_num_envs,), device=self.device, dtype=torch.float32)

        # Mask of touched links per env to deduplicate contacts without calling torch.unique
        self.env_link_touch = torch.zeros((self.total_num_envs, self.num_links), dtype=torch.bool, device=self.device)

    def compute_object_hand_contact(self):
        contact = self.object_hand_contact_buf
        contact_count = self.object_hand_contact_count_buf
        contact.zero_()
        contact_count.zero_()
        # clear env-link touch mask
        self.env_link_touch.zero_()

        num_contacts = self.gym.get_rigid_contacts(
            v.wrap_gpu_buffer(self.contact_normals_buf),
            v.wrap_gpu_buffer(self.contact_point_seps_buf),
            v.wrap_gpu_buffer(self.contact_id_a_buf),
            v.wrap_gpu_buffer(self.contact_id_b_buf),
            self.max_contact_pairs_per_env * self.total_num_envs,
        )

        num_stored = min(num_contacts, self.max_contact_pairs_per_env * self.total_num_envs)
        if num_stored <= 0:
            return
        id_a = self.contact_id_a_buf[:num_stored].to(torch.long)
        id_b = self.contact_id_b_buf[:num_stored].to(torch.long)

        env_a = self.contact_env_lookup[id_a[:, 1], id_a[:, 2]]
        env_b = self.contact_env_lookup[id_b[:, 1], id_b[:, 2]]
        same_env = env_a == env_b
        valid_env = torch.logical_and(env_a >= 0, env_b >= 0)
        valid_contact = torch.logical_and(same_env, valid_env)
        if not torch.any(valid_contact):
            return

        env_indices = env_a.clamp_min(0)

        object_indices = self.reward_object_transform_index_by_env[env_indices]
        a_is_object = id_a[:, 3] == object_indices
        b_is_object = id_b[:, 3] == object_indices

        hand_indices = self.hand_transform_indices_by_env[env_indices]
        monitored_hand_indices = hand_indices[:, self.monitored_link_mask]
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

            # Mark touched links per environment to deduplicate multiple contact pairs
            self.env_link_touch[contact_env_indices, hand_link_indices] = True

            # Which envs had any monitored-link touch
            touched_envs = torch.nonzero(self.env_link_touch.any(dim=1)).squeeze(-1)
            if touched_envs.numel() > 0:
                contact[touched_envs] = 1.0

            # Per-env unique contact counts = number of touched links per env
            contact_count[:] = self.env_link_touch.sum(dim=1).to(torch.float32)

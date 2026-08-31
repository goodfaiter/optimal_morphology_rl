from typing import List

import torch
import vlearn as v
from vlearn import gym

from optimal_morphology_rl.modules.object_generator import ObjectBase
from train.envs.environment import EnvironmentGpu


class Contacts:
    """Helper to compute object-hand contact metrics using environment buffers.

    This class keeps a reference to the environment instance and reads/writes
    the same GPU-backed tensors the env exposes. It also caches transform
    lookup tables that were previously computed on the environment.
    """

    def __init__(
        self,
        env: EnvironmentGpu,
        reward_object: ObjectBase,
        reward_object_link_name: str,
        link_names: List[str] | None = None,
        reward_object_link_offset: int | None = None,
    ) -> None:
        self.env: EnvironmentGpu = env
        self.device = self.env.device
        self.gym: gym.Gym = self.env.gym

        # Metadata
        self.max_contact_pairs_per_env = self.env.max_contact_pairs_per_env
        self.total_num_envs = self.env.total_num_envs
        self.num_links = self.env.robot.num_links

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
        self.hand_transform_indices_by_env = torch.full((self.total_num_envs, self.num_links), -1, dtype=torch.long, device=self.device)

        self.hand_transform_indices_by_env[:, :] = torch.arange(self.num_links, dtype=torch.long, device=self.device).unsqueeze(0)
        self.reward_object_transform_index_by_env[:] = self._compute_reward_object_transform_index(
            reward_object, reward_object_link_name, reward_object_link_offset
        )

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
                link_def = self.env.robot.art_def.get_link_def(i)
                if link_def.name.lower().endswith(name):
                    self.monitored_link_mask[i] = True

        if not torch.any(self.monitored_link_mask):
            raise ValueError("No monitored hand links were found.")

        # Output buffers owned by helper
        self.object_hand_contact_buf = torch.zeros((self.total_num_envs,), device=self.device, dtype=torch.float32)
        self.object_hand_contact_count_buf = torch.zeros((self.total_num_envs,), device=self.device, dtype=torch.float32)

        # Mask of touched links per env to deduplicate contacts without calling torch.unique
        self.env_link_touch = torch.zeros((self.total_num_envs, self.num_links), dtype=torch.bool, device=self.device)

    def _compute_reward_object_transform_index(
        self,
        reward_object: ObjectBase,
        reward_object_link_name: str,
        reward_object_link_offset: int | None = None,
    ) -> int:
        """Return the global transform-table index for the named reward-object link.

        The global transform table is laid out as:
            [hand links][object 0 links][object 1 links]...
        We compute the reward object's start offset from the cumulative link
        offsets and add the link's index within that object.

        Args:
            reward_object_link_offset: Pre-computed cumulative link offset for
                the reward object.  If ``None``, the offset is read from
                ``env.objects.get_object_link_offset`` for backward compatibility.
        """
        if reward_object_link_offset is None:
            reward_object_link_offset = self.env.objects.get_object_link_offset(
                reward_object.name
            )
        num_reward_object_links = reward_object.get_link_offset()
        start_offset = reward_object_link_offset - num_reward_object_links

        if hasattr(reward_object, "art_def") and reward_object.art_def is not None:
            art_def = reward_object.art_def
            link_index = None
            for i in range(art_def.get_num_link_defs()):
                if art_def.get_link_def(i).name == reward_object_link_name:
                    link_index = i
                    break
            if link_index is None:
                raise ValueError(
                    f"Reward object link '{reward_object_link_name}' not found in "
                    f"object '{reward_object.name}'. Available links: "
                    f"{[art_def.get_link_def(i).name for i in range(art_def.get_num_link_defs())]}"
                )
        else:
            # Rigid body: only one transform handle exists.
            link_index = 0

        return self.num_links + start_offset + link_index

    def update(self):
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

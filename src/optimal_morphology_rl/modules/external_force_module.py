"""Module that applies random external forces to the reward object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import vlearn as v

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@dataclass
class ExternalForceConfig:
    """Configuration for external force application."""

    apply_prob: float = 0.02
    force_max: float = 2.0
    torque_max: float = 0.0
    force_type: v.ForceType = v.ForceType.FORCE_TORQUE
    link_index: int = 0


def _uniform_sphere(n: int, device: torch.device) -> torch.Tensor:
    """Sample ``n`` directions uniformly distributed on the unit sphere."""
    phi = 2.0 * torch.pi * torch.rand(n, device=device)
    cos_theta = 2.0 * torch.rand(n, device=device) - 1.0
    sin_theta = torch.sqrt(1.0 - cos_theta**2)
    x = sin_theta * torch.cos(phi)
    y = sin_theta * torch.sin(phi)
    z = cos_theta
    return torch.stack([x, y, z], dim=-1)


class _BodyForceEntry:
    """Holds GPU buffer and command for a single body."""

    def __init__(
        self,
        name: str,
        handle: Any,
        num_envs: int,
        device: torch.device,
        config: ExternalForceConfig,
        env_group: Any,
        gym: v.Gym,
    ):
        self.name = name
        self.handle = handle
        self.num_envs = num_envs
        self.device = device
        self.config = config

        self.force_buf = torch.zeros((num_envs, 1, 6), dtype=torch.float32, device=device)
        self.cmd = env_group.create_rigid_body_external_force_command(
            v.wrap_gpu_buffer(self.force_buf),
            handle,
            force_type=config.force_type,
        )
        self.cmd_array = gym.create_gpu_array([self.cmd])

    def sample_and_apply(self, gym: v.Gym) -> None:
        """Sample and apply a random external force/torque."""
        apply_mask = torch.rand(self.num_envs, device=self.device) < self.config.apply_prob
        self.force_buf.zero_()

        if apply_mask.any():
            num_apply = apply_mask.sum().item()
            f_hi = self.config.force_max
            t_hi = self.config.torque_max

            random_forces = torch.zeros((num_apply, 6), device=self.device)
            force_mags = f_hi * torch.rand((num_apply, 1), device=self.device)
            random_forces[:, :3] = _uniform_sphere(num_apply, self.device) * force_mags
            torque_mags = t_hi * torch.rand((num_apply, 1), device=self.device)
            random_forces[:, 3:] = _uniform_sphere(num_apply, self.device) * torque_mags

            self.force_buf[apply_mask, 0, :] = random_forces

        gym.set_rigid_body_external_forces(self.cmd_array)


class _ExternalForceApplier:
    """Applies random external forces to a set of rigid bodies."""

    def __init__(
        self,
        body_handles: Dict[str, Any],
        total_num_envs: int,
        device: torch.device,
        env_group: Any,
        gym: v.Gym,
        config: Optional[ExternalForceConfig] = None,
    ):
        self.config = config or ExternalForceConfig()
        self._entries: Dict[str, _BodyForceEntry] = {
            name: _BodyForceEntry(
                name=name,
                handle=handle,
                num_envs=total_num_envs,
                device=device,
                config=self.config,
                env_group=env_group,
                gym=gym,
            )
            for name, handle in body_handles.items()
        }

    def step(self, gym: v.Gym) -> None:
        for entry in self._entries.values():
            entry.sample_and_apply(gym)


@register_module("external_force")
class ExternalForceModule(BaseModule):
    """Applies random external forces to the reward object.

    Only active when the reward object is a loaded rigid object that is not the
    cube (matching the legacy behavior).
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.apply_prob = float(self.config.get("apply_prob", 0.02))
        self.force_max = float(self.config.get("force_max", 2.0))
        self.torque_max = float(self.config.get("torque_max", 0.0))
        self.force_module: _ExternalForceApplier | None = None

    def post_finalize(self, container: ModuleContainer) -> None:
        """Create the external force applier if the reward object is eligible."""
        reward_object = container.get("reward_object")
        if reward_object is None:
            return

        from optimal_morphology_rl.modules.create_objects_module import LoadedRigidObject

        reward_object_name = container.get("reward_object_name", "")
        if reward_object_name == "cube" or not isinstance(reward_object, LoadedRigidObject):
            return

        config = ExternalForceConfig(
            apply_prob=self.apply_prob,
            force_max=self.force_max,
            torque_max=self.torque_max,
        )
        self.force_module = _ExternalForceApplier(
            body_handles={reward_object_name: reward_object.handle},
            total_num_envs=container.total_num_envs,
            device=container.device,
            env_group=container.env_group,
            gym=container.gym,
            config=config,
        )
        container.external_force = self.force_module

    def step(self, container: ModuleContainer) -> None:
        """Apply a random external force/torque to the reward object."""
        if self.force_module is not None:
            self.force_module.step(container.gym)

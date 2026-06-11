from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
import vlearn as v


@dataclass
class ExternalForceConfig:
    """Configuration for external force application."""

    # Probability of applying a force to any given body per step
    apply_prob: float = 0.02

    # Maximum force magnitude [N]
    force_max: float = 2.0

    # Maximum torque magnitude [Nm] — used for ForceType.FORCE_TORQUE
    torque_max: float = 0.0

    # Force type applied to all bodies
    force_type: v.ForceType = v.ForceType.FORCE_TORQUE

    # Link index within each body to apply the force to (0 = root link)
    link_index: int = 0


def _uniform_sphere(n: int, device: torch.device) -> torch.Tensor:
    """
    Sample ``n`` directions uniformly distributed on the unit sphere.

    Uses the standard spherical coordinate parameterisation:
        azimuth   phi   ~ Uniform[0, 2π)
        elevation theta ~ arccos(Uniform[-1, 1])   ← ensures equal solid-angle area

    Returns shape (n, 3).
    """
    phi = 2.0 * torch.pi * torch.rand(n, device=device)  # azimuth
    cos_theta = 2.0 * torch.rand(n, device=device) - 1.0  # cos(elevation)
    sin_theta = torch.sqrt(1.0 - cos_theta**2)

    x = sin_theta * torch.cos(phi)
    y = sin_theta * torch.sin(phi)
    z = cos_theta
    return torch.stack([x, y, z], dim=-1)


class BodyForceEntry:
    """Holds GPU buffer and command for a single body (rigid or articulated)."""

    def __init__(
        self,
        name: str,
        handle,
        num_envs: int,
        device: torch.device,
        config: ExternalForceConfig,
        env_group,
        gym: v.Gym,
    ):
        self.name = name
        self.handle = handle
        self.num_envs = num_envs
        self.device = device
        self.config = config

        # Buffer shape: (num_envs, 1, 6) — one link per body entry
        self.force_buf = torch.zeros((num_envs, 1, 6), dtype=torch.float32, device=device)

        self.cmd = env_group.create_rigid_body_external_force_command(
            v.wrap_gpu_buffer(self.force_buf),
            handle,
            force_type=config.force_type,
        )
        self.cmd_array = gym.create_gpu_array([self.cmd])

    def sample_and_apply(self, gym: v.Gym):
        """
        For each environment, independently decide (with probability apply_prob)
        whether to apply a new random force. Environments not selected keep their
        current force at zero (force is cleared each call for non-selected envs).
        """
        # Draw a per-environment Bernoulli mask
        apply_mask = torch.rand(self.num_envs, device=self.device) < self.config.apply_prob  # (N,)

        # Zero out all forces first so non-triggered envs receive no force
        self.force_buf.zero_()

        if apply_mask.any():
            num_apply = apply_mask.sum().item()
            f_hi = self.config.force_max
            t_hi = self.config.torque_max

            # Sample 6-DOF perturbation: [fx, fy, fz, tx/px, ty/py, tz/pz]
            # Forces and torques are sampled as a uniformly random direction on the
            # unit sphere scaled by a magnitude drawn uniformly from [0, max_magnitude].
            random_forces = torch.zeros((num_apply, 6), device=self.device)

            force_mags = f_hi * torch.rand((num_apply, 1), device=self.device)
            random_forces[:, :3] = _uniform_sphere(num_apply, self.device) * force_mags

            torque_mags = t_hi * torch.rand((num_apply, 1), device=self.device)
            random_forces[:, 3:] = _uniform_sphere(num_apply, self.device) * torque_mags

            self.force_buf[apply_mask, 0, :] = random_forces

        gym.set_rigid_body_external_forces(self.cmd_array)


class ExternalForceModule:
    """
    Applies random external forces to an explicit subset of rigid and/or
    articulated bodies.

    Bodies are supplied as named handle lists at construction time — only the
    bodies you pass in will ever receive perturbations, giving you full control
    over which objects are disturbed.

    Usage
    -----
        config = ExternalForceConfig(apply_prob=0.02)

        force_module = ExternalForceModule(
            rigid_bodies={"pen": pen_handle, "mug": mug_handle},
            articulated_bodies={"drawer": drawer_handle},
            total_num_envs=num_envs,
            device=device,
            env_group=env_group,
            gym=gym,
            config=config,
        )

        # Inside the simulation loop:
        while not done:
            force_module.step(gym)
            gym.step()
    """

    def __init__(
        self,
        body_handles: Dict[str, Any],
        total_num_envs: int,
        device: torch.device,
        env_group,
        gym: v.Gym,
        config: Optional[ExternalForceConfig] = None,
    ):
        """
        Parameters
        ----------
        body_handles:
            Mapping of name → body handle for bodies that should receive
            external forces. Pass an empty dict if none.
        total_num_envs:
            Total number of parallel environments.
        device:
            CUDA device for all GPU buffers.
        env_group:
            The environment group used to create force commands.
        gym:
            The gym instance.
        config:
            Force configuration. Defaults to ExternalForceConfig() if omitted.
        """
        self.config = config or ExternalForceConfig()

        self._entries: Dict[str, BodyForceEntry] = {
            name: BodyForceEntry(
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

    # ------------------------------------------------------------------
    # Step — call once per simulation step inside the loop
    # ------------------------------------------------------------------

    def step(self, gym: v.Gym) -> None:
        """
        For each registered body, independently sample (per environment) whether
        to apply a random external force this step.
        """
        for entry in self._entries.values():
            entry.sample_and_apply(gym)

    def __repr__(self) -> str:
        return (
            f"ExternalForceModule("
            f"apply_prob={self.config.apply_prob}, "
            f"force_type={self.config.force_type}, "
            f"bodies={list(self._entries.keys())})"
        )

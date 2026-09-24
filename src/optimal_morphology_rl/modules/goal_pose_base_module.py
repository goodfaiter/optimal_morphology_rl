"""Shared base for goal position and orientation sampling modules."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.helpers.numpy_vlearn import random_uniform_quaternion
from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer


class GoalPoseBaseModule(BaseModule):
    """Samples ``container.pos_goal_in_world`` ``(total_num_envs, 3)`` and
    ``container.quat_goal_to_world`` ``(total_num_envs, 4)`` (the quaternion
    rotates from the goal frame to the world frame); rewards and observations
    read these buffers.

    Goals are resampled for every env selected by the environment's reset
    buffer on every reset call (full ``env.reset()`` and mid-episode
    auto-resets). Subclasses define how goals are drawn between resets
    (discrete resampling or continuous sinusoids); see
    :class:`~optimal_morphology_rl.modules.goal_pose_discrete_module.GoalPoseDiscreteModule`
    and :class:`~optimal_morphology_rl.modules.goal_pose_sinusoidal_module.GoalPoseSinusoidalModule`.

    Config shape::

        goal_pose_discrete:
          goal_position:
            x: [0.0, 0.1]
            y: [-0.15, -0.15]
            z: [0.2, 0.3]
          goal_orientation: "random"  # or a fixed [x, y, z, w] quaternion
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        position_config = self.config.get("goal_position", {})
        x_range = position_config.get("x", [0.0, 0.0])
        y_range = position_config.get("y", [0.0, 0.0])
        z_range = position_config.get("z", [0.0, 0.0])
        self.x_range = (float(x_range[0]), float(x_range[1]))
        self.y_range = (float(y_range[0]), float(y_range[1]))
        self.z_range = (float(z_range[0]), float(z_range[1]))
        self.goal_orientation = self.config.get("goal_orientation", "random")

    def _allocate(self, container: ModuleContainer) -> None:
        """Allocate the goal buffers in the shared container."""
        total_num_envs = container.total_num_envs
        device = container.device
        container.pos_goal_in_world = torch.zeros((total_num_envs, 3), device=device, dtype=torch.float32)
        container.quat_goal_to_world = torch.zeros((total_num_envs, 4), device=device, dtype=torch.float32)

    def post_finalize(self, container: ModuleContainer) -> None:
        """Allocate the goal buffers once the container is populated."""
        if "pos_goal_in_world" not in container:
            self._allocate(container)

    def reset(self, container: ModuleContainer) -> None:
        """Resample the goal for the envs selected by the environment's reset buffer."""
        if "pos_goal_in_world" not in container:
            if container.get("total_num_envs") is None or container.get("device") is None:
                raise RuntimeError(
                    "goal_pose modules require 'total_num_envs' and 'device' in the shared "
                    "container. Ensure create_rigid_vsim_envs is listed before them."
                )
            self._allocate(container)

        # The goals change on every reset call: all envs on env.reset() and the
        # reset buffer's envs on mid-episode auto-resets.
        reset_buf = container.get("reset_buf")
        mask = None if reset_buf is None else reset_buf
        self._resample(container, mask)

    def _resample(self, container: ModuleContainer, mask: torch.Tensor | None) -> None:
        """Draw the goal for the envs selected by ``mask`` (all if None)."""
        raise NotImplementedError

    def _sample(self, pos_buf: torch.Tensor, quat_buf: torch.Tensor, mask: torch.Tensor | None) -> None:
        """Sample uniform positions and configured orientations for the envs selected by ``mask``."""
        num = pos_buf.shape[0] if mask is None else int(mask.sum().item())
        device = pos_buf.device

        pos = torch.stack(
            [
                self.x_range[0] + (self.x_range[1] - self.x_range[0]) * torch.rand(num, device=device),
                self.y_range[0] + (self.y_range[1] - self.y_range[0]) * torch.rand(num, device=device),
                self.z_range[0] + (self.z_range[1] - self.z_range[0]) * torch.rand(num, device=device),
            ],
            dim=-1,
        )
        self._sample_quat(quat_buf, mask)

        if mask is None:
            pos_buf[:] = pos
        else:
            pos_buf[mask] = pos

    def _sample_quat(self, quat_buf: torch.Tensor, mask: torch.Tensor | None) -> None:
        """Sample orientations for the envs selected by ``mask`` (all if None)."""
        num = quat_buf.shape[0] if mask is None else int(mask.sum().item())
        if self.goal_orientation == "random":
            quat = random_uniform_quaternion(num, device=quat_buf.device, dtype=torch.float32)
        else:
            quat = torch.tensor(self.goal_orientation, device=quat_buf.device, dtype=torch.float32).expand(num, 4)
        if mask is None:
            quat_buf[:] = quat
        else:
            quat_buf[mask] = quat

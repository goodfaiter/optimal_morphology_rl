"""Module that owns goal position and orientation sampling for the environment."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.helpers.numpy_vlearn import random_uniform_quaternion
from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("goal_pose")
class GoalPoseModule(BaseModule):
    """Samples goal positions and orientations into the shared container.

    Writes ``container.pos_goal_in_world`` ``(total_num_envs, 3)`` and
    ``container.quat_goal_to_world`` ``(total_num_envs, 4)`` (the
    quaternion rotates from the goal frame to the world frame); rewards and
    observations read these buffers.

    Goals are resampled for every env selected by the environment's reset
    buffer on every reset call (full ``env.reset()`` and mid-episode
    auto-resets), and periodically resampled for random envs every step
    (``resample_prob`` per env per step).

    Config shape::

        goal_pose:
          goal_position:
            x: [-0.09, 0.0]
            y: [-0.12, -0.12]
            z: [0.15, 0.20]
          goal_orientation: "random"  # or a fixed [x, y, z, w] quaternion
          resample_prob: 0.0
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
        self.resample_prob = float(self.config.get("resample_prob", 0.0))

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
                    "goal_pose requires 'total_num_envs' and 'device' in the shared "
                    "container. Ensure create_rigid_vsim_envs is listed before goal_pose."
                )
            self._allocate(container)

        # The goals change on every reset call: all envs on env.reset() and the
        # reset buffer's envs on mid-episode auto-resets.
        reset_buf = container.get("reset_buf")
        mask = None if reset_buf is None else reset_buf
        self._sample(container.pos_goal_in_world, container.quat_goal_to_world, mask)

    def step(self, container: ModuleContainer) -> None:
        """Periodically resample the goal for random envs (``resample_prob``)."""
        if self.resample_prob <= 0.0:
            return
        if "pos_goal_in_world" not in container:
            return
        mask = torch.rand(container.total_num_envs, device=container.device) < self.resample_prob
        if mask.any():
            self._sample(container.pos_goal_in_world, container.quat_goal_to_world, mask)

    def _sample(self, pos_buf: torch.Tensor, quat_buf: torch.Tensor, mask: torch.Tensor | None) -> None:
        """Sample the goal for the envs selected by ``mask`` (all if None)."""
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
        if self.goal_orientation == "random":
            quat = random_uniform_quaternion(num, device=device, dtype=torch.float32)
        else:
            quat = torch.tensor(self.goal_orientation, device=device, dtype=torch.float32).expand(num, 4)

        if mask is None:
            pos_buf[:] = pos
            quat_buf[:] = quat
        else:
            pos_buf[mask] = pos
            quat_buf[mask] = quat

"""Goal-pose module that samples goals discretely: on resets and via ``resample_prob``."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.goal_pose_base_module import GoalPoseBaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("goal_pose_discrete")
class GoalPoseDiscreteModule(GoalPoseBaseModule):
    """Samples goal positions and orientations into the shared container.

    Positions are drawn uniformly from the ``goal_position`` ranges and
    orientations from ``goal_orientation`` (random quaternion or fixed
    ``[x, y, z, w]``). Goals change on every reset call (full ``env.reset()``
    and mid-episode auto-resets for the reset buffer's envs), and are
    additionally periodically resampled for random envs every step
    (``resample_prob`` per env per step).

    Config shape::

        goal_pose_discrete:
          goal_position:
            x: [0.0, 0.1]
            y: [-0.15, -0.15]
            z: [0.2, 0.3]
          goal_orientation: "random"  # or a fixed [x, y, z, w] quaternion
          resample_prob: 0.0
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.resample_prob = float(self.config.get("resample_prob", 0.0))

    def _resample(self, container: ModuleContainer, mask: torch.Tensor | None) -> None:
        """Resample the static goal for the envs selected by ``mask`` (all if None)."""
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

"""Observation source for a fingertip-reach goal sampled in the world frame."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.observations.observation_base_module import (
    ObservationBaseModule,
)
from optimal_morphology_rl.modules.observations.observation_manager_module import (
    register_observation,
)


@register_observation("goal_finger_tip")
class GoalFingerTipObservation(ObservationBaseModule):
    """Fingertip-reach goal observation.

    Reads ``container.pos_goal_in_world`` ``(total_num_envs, 3)``, which is
    owned and sampled by the ``goal_pose_discrete`` / ``goal_pose_sinusoidal``
    / ``goal_pose_square`` modules. This module only observes the goal; goals
    change on every reset call, and in continuous modes they move every step.
    """

    def get_obs_dim(self, env: Any) -> int:
        return 3

    def compute_observation(self, env: Any, out: torch.Tensor) -> None:
        container = env.module_manager.container
        goal = container.get("pos_goal_in_world")
        if goal is None:
            raise RuntimeError(
                "goal_finger_tip: container.pos_goal_in_world is missing. "
                "Add a goal_pose_discrete, goal_pose_sinusoidal, or goal_pose_square module to post_physics_step_modules before observation."
            )
        out[:] = goal

"""Observation source for a fingertip-reach goal, in the goal_pose goal frame."""

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

    Reads ``container.goal_position_in_goal_frame`` ``(total_num_envs, 3)``, the goal in the
    ``goal_pose`` goal frame (``goal_pose.frame``, e.g. Pinky_Abd; the environment frame without one),
    i.e. within the configured ``goal_position`` ranges and measurable on the real hand. This module
    only observes the goal; goals change on every reset call, and the moving goal types move every step.
    """

    def get_obs_dim(self, env: Any) -> int:
        return 3

    def compute_observation(self, env: Any, out: torch.Tensor) -> None:
        container = env.module_manager.container
        goal = container.get("goal_position_in_goal_frame")
        if goal is None:
            raise RuntimeError(
                "goal_finger_tip: container.goal_position_in_goal_frame is missing. "
                "Add the goal_pose module to post_physics_step_modules before observation."
            )
        out[:] = goal

"""Observation source for the object goal pose in the robot frame."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.observations.observation_jit_helpers import (
    _goal_state_obs_jit,
)
from optimal_morphology_rl.modules.observations.observation_base_module import (
    ObservationBaseModule,
)
from optimal_morphology_rl.modules.observations.observation_manager_module import (
    register_observation,
)


@register_observation("goal_state")
class GoalStateObservation(ObservationBaseModule):
    """Goal position and orientation expressed in the robot base frame."""

    def get_obs_dim(self, env: Any) -> int:
        return 3 + 6  # goal pos, 6d goal rot

    def compute_observation(self, env: Any, out: torch.Tensor) -> None:
        container = env.module_manager.container
        goal_pos = container.get("pos_goal_in_world")
        goal_quat = container.get("quat_goal_to_world")
        if goal_pos is None or goal_quat is None:
            raise RuntimeError(
                "goal_state: container.pos_goal_in_world / quat_goal_to_world are missing. "
                "Add the goal_pose module to post_physics_step_modules before observation."
            )
        robot_state = container.robot_state

        out[:] = _goal_state_obs_jit(
            robot_state["quat_robot_to_world"],
            robot_state["robot_pos_in_world"],
            goal_pos,
            goal_quat,
        )

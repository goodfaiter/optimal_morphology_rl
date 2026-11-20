"""Observation source for the object goal pose in the robot frame."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.goal_pose_module import goal_in_env_frame
from optimal_morphology_rl.modules.observations.observation_base_module import (
    ObservationBaseModule,
)
from optimal_morphology_rl.modules.observations.observation_jit_helpers import (
    _goal_state_obs_jit,
)
from optimal_morphology_rl.modules.observations.observation_manager_module import (
    register_observation,
)

_SIGNALS = {"pos": slice(0, 3), "rot": slice(3, 9)}


@register_observation("goal_state")
class GoalStateObservation(ObservationBaseModule):
    """Goal position and orientation expressed in the robot base frame.

    ``signals`` (optional, default all) selects and orders the parts: ``pos`` (m, 3) and ``rot`` (6D rotation, 6)::

        goal_state:
          signals: [pos]
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.signals = list(self.config.get("signals", list(_SIGNALS)))
        unknown = [s for s in self.signals if s not in _SIGNALS]
        if unknown or not self.signals:
            raise RuntimeError(f"goal_state observation: 'signals' must be a non-empty subset of {list(_SIGNALS)}, got {self.signals}.")

    def get_obs_dim(self, env: Any) -> int:
        return sum(_SIGNALS[s].stop - _SIGNALS[s].start for s in self.signals)

    def compute_observation(self, env: Any, out: torch.Tensor) -> None:
        container = env.module_manager.container
        # Relative to the robot, so it needs the goal in the environment frame.
        goal_pos, goal_quat = goal_in_env_frame(container)
        robot_state = container.robot_state

        full = _goal_state_obs_jit(
            robot_state["quat_robot_to_world"],
            robot_state["robot_pos_in_world"],
            goal_pos,
            goal_quat,
        )
        out[:] = torch.cat([full[:, _SIGNALS[s]] for s in self.signals], dim=-1)

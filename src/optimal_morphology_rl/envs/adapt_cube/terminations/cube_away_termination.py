"""Termination when the cube leaves the goal region (e.g. rolls off the palm without dropping below the drop height)."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.create_kinematic_sensor_module import (
    find_kinematic_sensor,
)
from optimal_morphology_rl.modules.goal_pose_module import goal_in_env_frame
from optimal_morphology_rl.modules.terminations.termination_base_module import (
    TerminationBaseModule,
    register_termination,
)


@register_termination("cube_away_termination")
class CubeAwayTermination(TerminationBaseModule):
    """Terminates episodes when the reward object is farther than ``max_distance`` (m) from the goal position.

    Config::

        cube_away_termination:
          max_distance: 0.04
          reward_scale: -200.0   # added once on termination

    Logs ``away_rate``: the share of the last games_to_track ended episodes that ended with the object away
    (see TerminationBaseModule.rate_name).
    """

    rate_name = "away_rate"

    def compute(self, env: Any) -> None:
        container = env.module_manager.container
        sensor = find_kinematic_sensor(container, container.reward_object.name)
        if sensor is None:
            raise RuntimeError("cube_away_termination: no kinematic sensor for the reward object.")
        goal_pos, _ = goal_in_env_frame(container)
        max_distance = float(self.config.get("max_distance", 0.04))
        reward_scale = float(self.config.get("reward_scale", -200.0))

        away = torch.norm(sensor.pos_in_world - goal_pos, dim=-1) > max_distance
        env.term_buf[:] = torch.logical_or(env.term_buf, away)
        if reward_scale != 0.0:
            env.rew_buf[:] += reward_scale * away.float()
        self.triggered = away

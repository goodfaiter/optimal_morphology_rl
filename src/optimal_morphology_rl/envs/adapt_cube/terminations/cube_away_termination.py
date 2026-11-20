"""Termination when the cube leaves its allowed region: too far from the goal and / or outside a box (e.g. around the fingers)."""

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
    """Terminates episodes when the reward object leaves its allowed region.

    Two checks, each optional (at least one is required), either one terminates:

    - ``max_distance`` (m): farther than this from the goal position.
    - ``box``: outside an axis-aligned box in the environment frame, ``{x: [lo, hi], y: [lo, hi], z: [lo, hi]}``
      (an axis left out is unbounded), e.g. around the fingers so that falling off the hand, onto the palm or being
      thrown up ends the episode.

    Config::

        cube_away_termination:
          max_distance: 0.04
          box: {x: [-0.055, 0.035], y: [-0.23, -0.10], z: [0.115, 0.22]}
          reward_scale: -200.0   # added once on termination

    Logs ``away_rate``: the share of the last games_to_track ended episodes that ended with the object away
    (see TerminationBaseModule.rate_name).
    """

    rate_name = "away_rate"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.max_distance = self.config.get("max_distance")
        box = self.config.get("box")
        if self.max_distance is None and box is None:
            raise ValueError("cube_away_termination needs 'max_distance' and / or 'box'.")
        self.box: list[tuple[int, float, float]] = []
        for axis, key in enumerate("xyz"):
            if box is not None and box.get(key) is not None:
                lo, hi = float(box[key][0]), float(box[key][1])
                if lo >= hi:
                    raise ValueError(f"cube_away_termination box {key}: lo must be < hi, got {[lo, hi]}.")
                self.box.append((axis, lo, hi))

    def compute(self, env: Any) -> None:
        container = env.module_manager.container
        sensor = find_kinematic_sensor(container, container.reward_object.name)
        if sensor is None:
            raise RuntimeError("cube_away_termination: no kinematic sensor for the reward object.")
        reward_scale = float(self.config.get("reward_scale", -200.0))
        pos = sensor.pos_in_world

        away = torch.zeros(pos.shape[0], dtype=torch.bool, device=pos.device)
        if self.max_distance is not None:
            goal_pos, _ = goal_in_env_frame(container)
            away |= torch.norm(pos - goal_pos, dim=-1) > float(self.max_distance)
        for axis, lo, hi in self.box:
            away |= (pos[:, axis] < lo) | (pos[:, axis] > hi)
        env.term_buf[:] = torch.logical_or(env.term_buf, away)
        if reward_scale != 0.0:
            env.rew_buf[:] += reward_scale * away.float()
        self.triggered = away

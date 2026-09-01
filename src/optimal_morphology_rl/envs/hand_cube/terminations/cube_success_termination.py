"""Termination when the cube orientation success condition is met."""

from __future__ import annotations

import math
from typing import Any

import torch

from optimal_morphology_rl.envs.hand_envs.utils import get_reward_object_name
from optimal_morphology_rl.modules.terminations.termination_base_module import (
    TerminationBaseModule,
    register_termination,
)


@register_termination("cube_success_termination")
class CubeSuccessTermination(TerminationBaseModule):
    """Terminates cube episodes when the orientation stays aligned long enough."""

    def post_finalize(self, env: Any) -> None:
        self.goal_aligned_steps_buf = torch.zeros(
            (env.total_num_envs,), device=env.device, dtype=torch.long
        )

    def reset(self, env: Any) -> None:
        self.goal_aligned_steps_buf[env.reset_buf] = 0

    def compute(self, env: Any) -> None:
        if get_reward_object_name(env) != "cube":
            return

        container = env.module_manager.container
        reward_object = container.reward_object
        quat_object_to_world = container.kinematic_sensor.quat_sensor_to_world
        quat_object_goal_to_world = reward_object.goal_quat_object_to_world

        quat_dot = (
            torch.sum(quat_object_goal_to_world * quat_object_to_world, dim=-1)
            .abs()
            .clamp(max=1.0)
        )
        goal_angle = 2.0 * torch.acos(quat_dot)
        is_aligned = goal_angle < math.radians(15.0)

        self.goal_aligned_steps_buf[is_aligned] += 1
        self.goal_aligned_steps_buf[~is_aligned] = 0
        goal_success = self.goal_aligned_steps_buf > 30

        env.term_buf[:] = torch.logical_or(env.term_buf, goal_success)

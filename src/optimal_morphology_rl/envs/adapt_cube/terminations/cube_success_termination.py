"""Termination and success bonus for the cube orientation success condition."""

from __future__ import annotations

import math
from typing import Any

import torch
from time_series_buffer.time_series_buffer import TimeSeriesBuffer

from optimal_morphology_rl.envs.hand_envs.utils import get_reward_object_name
from optimal_morphology_rl.modules.terminations.termination_base_module import (
    TerminationBaseModule,
    register_termination,
)


@register_termination("cube_success_termination")
class CubeSuccessTermination(TerminationBaseModule):
    """Terminates cube episodes when the orientation stays aligned long enough.

    Adds the large success bonus to ``env.rew_buf`` and reports the cumulative
    goal success rate over the last 1000 steps, matching the legacy
    ``compute_reward_termination_truncation`` behavior.
    """

    def post_finalize(self, env: Any) -> None:
        self.goal_aligned_steps_buf = torch.zeros((env.total_num_envs,), device=env.device, dtype=torch.long)
        self.episode_success_history = TimeSeriesBuffer(
            num_envs=1,
            dim=2,
            max_size=1000,
            stride=1,
            device=env.device,
        )
        self.episode_success_history_count = 0

    def reset(self, env: Any) -> None:
        self.goal_aligned_steps_buf[env.reset_buf] = 0

    def compute(self, env: Any) -> None:
        if get_reward_object_name(env) != "cube":
            return

        container = env.module_manager.container
        reward_object = container.reward_object
        quat_object_to_world = container.kinematic_sensor.quat_sensor_to_world
        quat_object_goal_to_world = reward_object.goal_quat_object_to_world

        quat_dot = torch.sum(quat_object_goal_to_world * quat_object_to_world, dim=-1).abs().clamp(max=1.0)
        goal_angle = 2.0 * torch.acos(quat_dot)
        is_aligned = goal_angle < math.radians(15.0)

        self.goal_aligned_steps_buf[is_aligned] += 1
        self.goal_aligned_steps_buf[~is_aligned] = 0
        goal_success = self.goal_aligned_steps_buf > 30

        success_scale = float(self.config.get("success_scale", 200.0))
        env.rew_buf[:] += success_scale * goal_success.float()
        env.term_buf[:] = torch.logical_or(env.term_buf, goal_success)

        # Cumulative goal success rate over the last 1000 env steps.
        episode_end = torch.logical_or(env.term_buf, env.trunc_buf)
        num_ended = episode_end.sum()
        if num_ended > 0:
            num_success = goal_success[episode_end].float().sum()
            entry = torch.stack([num_success, num_ended.float()], dim=-1).unsqueeze(0)
        else:
            entry = torch.zeros((1, 2), device=env.device, dtype=torch.float32)

        self.episode_success_history.add(entry)
        self.episode_success_history_count = min(self.episode_success_history_count + 1, 1000)

        history = self.episode_success_history.get()[0]
        filled_history = history[: self.episode_success_history_count]
        total_ended = filled_history[:, 1].sum()
        if total_ended > 0:
            total_success = filled_history[:, 0].sum()
            env.info["rewards"]["goal_success_rate"] = (total_success / total_ended).item()

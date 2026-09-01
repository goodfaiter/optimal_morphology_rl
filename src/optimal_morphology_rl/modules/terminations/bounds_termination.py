"""Termination when the robot hand leaves the table bounds."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.terminations.termination_base_module import (
    TerminationBaseModule,
    register_termination,
)


@register_termination("bounds_termination")
class BoundsTermination(TerminationBaseModule):
    """Terminates episodes when the robot base leaves the configured table area."""

    def post_finalize(self, container: ModuleContainer) -> None:
        device = container.device

        table_obj = container.objects.get("table") or container.objects.get("table_with_camera")
        if table_obj is None:
            raise RuntimeError("BoundsTermination requires a 'table' or 'table_with_camera' object.")
        table_half_size = table_obj.half_size
        padding = float(self.config.get("padding", 0.2))
        self.table_bounds = torch.tensor(
            [
                [-table_half_size.x - padding, table_half_size.x + padding],
                [-table_half_size.y - padding, table_half_size.y + padding],
                [0.0, 0.35],
            ],
            device=device,
            dtype=torch.float32,
        )

    def compute(self, env: Any) -> None:
        robot = env.module_manager.container.robot
        reward_scale = float(self.config.get("reward_scale", -10.0))

        out_of_bounds = torch.logical_or(
            robot.robot_pos_in_world < self.table_bounds[:, 0],
            robot.robot_pos_in_world > self.table_bounds[:, 1],
        )
        bounds = torch.any(out_of_bounds, dim=-1)
        env.term_buf[:] = torch.logical_or(env.term_buf, bounds)

        env.rew_buf[:] += reward_scale * bounds.float()
        env.info["rewards"]["bounds_penalty"] = -bounds.float().sum().item() / env.total_num_envs

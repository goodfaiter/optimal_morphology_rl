"""Termination when the reward object drops below a height threshold."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.create_kinematic_sensor_module import (
    find_kinematic_sensor,
)
from optimal_morphology_rl.modules.terminations.termination_base_module import (
    TerminationBaseModule,
    register_termination,
)


@register_termination("drop_termination")
class DropTermination(TerminationBaseModule):
    """Terminates episodes when the object falls below a z threshold."""

    def compute(self, env: Any) -> None:
        container = env.module_manager.container
        sensor = find_kinematic_sensor(container, container.reward_object.name)
        if sensor is None:
            raise RuntimeError(
                "drop_termination: no kinematic sensor for the reward object. "
                "Ensure create_kinematic_sensor is listed."
            )
        object_pos_in_world = sensor.pos_in_world
        threshold = float(self.config.get("threshold", -0.1))
        reward_scale = float(self.config.get("reward_scale", -20.0))

        drop = object_pos_in_world[:, 2] < threshold
        env.term_buf[:] = torch.logical_or(env.term_buf, drop)

        if reward_scale != 0.0:
            env.rew_buf[:] += reward_scale * drop.float()
            env.info["rewards"]["drop_penalty"] = -drop.float().sum().item() / env.total_num_envs

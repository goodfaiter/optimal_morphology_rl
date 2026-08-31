"""Termination when the reward object drops below a height threshold."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.terminations.termination_base_module import (
    TerminationBaseModule,
)
from optimal_morphology_rl.modules.terminations.termination_manager_module import (
    register_termination,
)


@register_termination("drop_termination")
class DropTermination(TerminationBaseModule):
    """Terminates episodes when the object falls below a z threshold."""

    def compute(self, env: Any) -> None:
        container = env.module_manager.container
        object_pos_in_world = container.kinematic_sensor.pos_in_world
        threshold = float(self.config.get("threshold", -0.1))

        drop = object_pos_in_world[:, 2] < threshold
        env.term_buf[:] = torch.logical_or(env.term_buf, drop)

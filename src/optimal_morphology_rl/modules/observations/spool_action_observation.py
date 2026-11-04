"""Observation source for the policy's current spool command (the action of this step)."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.observations.observation_base_module import (
    ObservationBaseModule,
)
from optimal_morphology_rl.modules.observations.observation_manager_module import (
    register_observation,
)


@register_observation("spool_action")
class SpoolActionObservation(ObservationBaseModule):
    """The scaled policy action per spool of the step just taken (rad, the commanded change of the spool angle:
    desired = position + action), from 'create_spool' (``commanded_delta_angle``); 0 after a reset. With the
    observation history (``num_hist``) the policy sees its recent commands, which the robot knows too.

    Layout: ``[action (num_model_tendons)]``.

    Config shape (under ``observation.modules``)::

        spool_action: {}
    """

    def _spool(self, env: Any):
        spool = env.module_manager.container.get("spool")
        if spool is None:
            raise RuntimeError("spool_action observation requires the 'create_spool' module.")
        return spool

    def get_obs_dim(self, env: Any) -> int:
        return len(self._spool(env).model_tendon_indices)

    def compute_observation(self, env: Any, out: torch.Tensor) -> None:
        out[:] = self._spool(env).commanded_delta_angle(env.module_manager.container)

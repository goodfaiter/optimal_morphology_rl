"""Base class for observation sub-modules managed by ObservationManagerModule."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import torch

from optimal_morphology_rl.modules.base_module import BaseModule


class ObservationBaseModule(BaseModule):
    """Single observation source that writes a slice of the base observation.

    Subclasses implement :meth:`get_obs_dim` and :meth:`compute_observation`.
    The manager is responsible for concatenating the slices, applying history,
    and writing the result to ``env.obs_buf``.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

    @abstractmethod
    def get_obs_dim(self, env: Any) -> int:
        """Return the flat observation dimension produced by this module."""
        raise NotImplementedError

    @abstractmethod
    def compute_observation(self, env: Any, out: torch.Tensor) -> None:
        """Write this module's observation into ``out``.

        Args:
            env: The environment instance.
            out: Tensor of shape ``(total_num_envs, obs_dim)`` to populate.
        """
        raise NotImplementedError

    def reset(self, env: Any) -> None:
        """Optional reset hook."""
        pass

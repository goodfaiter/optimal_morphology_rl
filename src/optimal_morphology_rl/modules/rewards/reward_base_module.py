"""Base class for reward sub-modules managed by RewardManagerModule."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from optimal_morphology_rl.modules.base_module import BaseModule


class RewardBaseModule(BaseModule):
    """Single reward source.

    Subclasses implement :meth:`compute` and add to ``env.rew_buf`` and
    ``env.info["rewards"]``.  Termination/truncation is handled by the
    separate termination manager.
    """

    @abstractmethod
    def compute(self, env: Any) -> None:
        """Compute this module's contribution to the reward."""
        raise NotImplementedError

    def finalize(self, env: Any) -> None:
        """Optional finalize hook."""
        pass

    def post_finalize(self, env: Any) -> None:
        """Optional post-finalize hook for buffer allocation."""
        pass

    def reset(self, env: Any) -> None:
        """Optional reset hook."""
        pass

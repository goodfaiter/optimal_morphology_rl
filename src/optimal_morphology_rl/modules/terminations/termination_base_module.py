"""Base class for termination sub-modules managed by TerminationManagerModule."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from optimal_morphology_rl.modules.base_module import BaseModule


class TerminationBaseModule(BaseModule):
    """Single termination source.

    Subclasses implement :meth:`compute` and write to ``env.term_buf``.
    Truncation is handled by the termination manager.
    """

    @abstractmethod
    def compute(self, env: Any) -> None:
        """Compute this module's contribution to the termination buffer."""
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

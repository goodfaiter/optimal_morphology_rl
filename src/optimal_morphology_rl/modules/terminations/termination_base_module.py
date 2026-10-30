"""Base class for termination sub-modules managed by TerminationManagerModule."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, TypeVar

import torch

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer


T = TypeVar("T", bound="TerminationBaseModule")

# Registry for termination sub-modules.
TERMINATION_REGISTRY: dict[str, type[TerminationBaseModule]] = {}


def register_termination(name: str | None = None) -> Callable[[type[T]], type[T]]:
    """Decorator that registers a termination sub-module class."""

    def decorator(cls: type[T]) -> type[T]:
        registry_name = name if name is not None else cls.__name__
        if registry_name in TERMINATION_REGISTRY:
            raise ValueError(f"Termination module '{registry_name}' is already registered.")
        TERMINATION_REGISTRY[registry_name] = cls
        return cls

    return decorator


class TerminationBaseModule(BaseModule):
    """Single termination or truncation source.

    Subclasses implement :meth:`compute` and write to ``env.term_buf`` and/or
    ``env.trunc_buf``.  The termination manager aggregates both into
    ``container.reset_buf``.

    A subclass that sets ``rate_name`` (e.g. ``drop_rate``) and stores the envs it ended this step in
    ``self.triggered`` (in :meth:`compute`) gets that rate logged in ``env.info["rewards"]``: the share of the last
    ``num_episodes`` ended episodes (the agent's ``games_to_track``, as its mean reward) that it ended. The manager
    updates it after every termination has run, so episode ends of any termination count, whatever their order.
    """

    rate_name: str | None = None
    triggered: torch.Tensor | None = None

    @abstractmethod
    def compute(self, env: Any) -> None:
        """Compute this module's contribution to the termination buffers."""
        raise NotImplementedError

    def log_end_rate(self, env: Any, episode_end: torch.Tensor, num_episodes: int = 100) -> None:
        """Log ``rate_name``: this module's share of the last ``num_episodes`` ended episodes."""
        if self.rate_name is None or self.triggered is None:
            return
        if getattr(self, "_outcomes", None) is None:
            # Ring buffer over the last ended episodes: 1 if this module ended it, else 0.
            self._outcomes = torch.zeros(num_episodes, device=episode_end.device, dtype=torch.float32)
            self._num_outcomes = 0
            self._next_outcome = 0
        size = self._outcomes.numel()
        flags = self.triggered[episode_end].float()[-size:]
        if flags.numel() > 0:
            index = (self._next_outcome + torch.arange(flags.numel(), device=flags.device)) % size
            self._outcomes[index] = flags
            self._next_outcome = (self._next_outcome + flags.numel()) % size
            self._num_outcomes = min(self._num_outcomes + flags.numel(), size)
        if self._num_outcomes > 0:
            env.info["rewards"][self.rate_name] = self._outcomes[: self._num_outcomes].mean().item()

    def finalize(self, container: ModuleContainer) -> None:
        """Optional finalize hook."""
        pass

    def post_finalize(self, container: ModuleContainer) -> None:
        """Optional post-finalize hook for buffer allocation."""
        pass

    def reset(self, container: ModuleContainer) -> None:
        """Optional reset hook."""
        pass

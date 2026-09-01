"""Base class for termination sub-modules managed by TerminationManagerModule."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, TypeVar

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
    """

    @abstractmethod
    def compute(self, env: Any) -> None:
        """Compute this module's contribution to the termination buffers."""
        raise NotImplementedError

    def finalize(self, container: ModuleContainer) -> None:
        """Optional finalize hook."""
        pass

    def post_finalize(self, container: ModuleContainer) -> None:
        """Optional post-finalize hook for buffer allocation."""
        pass

    def reset(self, container: ModuleContainer) -> None:
        """Optional reset hook."""
        pass

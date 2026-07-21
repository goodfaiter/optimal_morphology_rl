"""Base class for environment modules used by :class:`ModuleManager`."""

from abc import ABC
from typing import Any


class BaseModule(ABC):
    """Minimal lifecycle interface for a modular environment component.

    Subclasses override only the hooks they need.  The manager dispatches each
    hook to every registered module in the order given by the YAML ``modules``
    list.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the module with its own config slice.

        Args:
            config: The configuration dictionary belonging to this module
                (e.g. the value of ``config["robot"]``).  ``None`` is treated
                as an empty dict.
        """
        self.config = config or {}

    def finalize(self, env: Any) -> None:
        """Called once after all modules have been instantiated.

        Use this hook to resolve cross-module dependencies via
        ``env.module_manager`` or ``env.modules``.

        Args:
            env: The environment instance that owns this manager.
        """
        pass

    def post_finalize(self, env: Any) -> None:
        """Called once after :meth:`finalize` has run for every module.

        Use this for construction steps that require the shared container to
        be fully populated by earlier finalize hooks.
        """
        pass

    def pre_physics_step(self, env: Any) -> None:
        """Called before the physics step with the current actions applied."""
        pass

    def pre_gym_step(self, env: Any) -> None:
        """Called at the beginning of each simulation sub-step."""
        pass

    def post_gym_step(self, env: Any) -> None:
        """Called at the end of each simulation sub-step."""
        pass

    def post_physics_step(self, env: Any) -> None:
        """Called after the physics step (compute obs/reward/reset)."""
        pass

    def reset(self, env: Any) -> None:
        """Reset the modules selected by the environment's reset buffer.

        This maps to the existing ``reset_idx`` convention.
        """
        pass

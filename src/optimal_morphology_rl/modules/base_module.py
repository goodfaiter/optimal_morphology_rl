"""Base class for environment modules used by :class:`ModuleManager`."""

from abc import ABC
from typing import Any

from optimal_morphology_rl.modules.module_container import ModuleContainer


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

    def finalize(self, container: ModuleContainer) -> None:
        """Called once after all modules have been instantiated.

        Use this hook to resolve cross-module dependencies via
        ``env.module_manager`` or ``env.modules``.

        Args:
            container: The module container.
        """
        pass

    def post_finalize(self, container: ModuleContainer) -> None:
        """Called once after :meth:`finalize` has run for every module.

        Use this for construction steps that require the shared container to
        be fully populated by earlier finalize hooks.
        """
        pass

    def step(self, container: ModuleContainer) -> None:
        """Main step function called at pre/post gym/physics steps."""
        pass

    def reset(self, container: ModuleContainer) -> None:
        """Reset the modules selected by the environment's reset buffer.

        This maps to the existing ``reset_idx`` convention.
        """
        pass

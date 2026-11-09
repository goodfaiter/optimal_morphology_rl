"""Base class for environment modules used by :class:`ModuleManager`."""

from abc import ABC
from typing import Any

from optimal_morphology_rl.modules.module_container import ModuleContainer


def curriculum_range(low: Any, high: Any, factor: float, nominal: Any = None) -> tuple[Any, Any]:
    """A randomization range ``[low, high]`` shrunk toward ``nominal`` (default: its midpoint) by the training
    curriculum factor: just the nominal at 0, the full range at 1. Works on floats and tensors (per-element nominal)."""
    if nominal is None:
        nominal = (low + high) / 2
    return nominal + factor * (low - nominal), nominal + factor * (high - nominal)


def seconds_to_steps(seconds: float, step_dt: float, name: str, exact: bool = True) -> int:
    """A duration (s) as a whole number of steps of ``step_dt`` (s), e.g. control steps (``container.control_dt``), so
    configs in seconds hold for any frame_skip. ``exact``: raise unless the duration is a whole number of steps (within
    1e-6), e.g. a history spacing; otherwise round to the nearest step (at least 1)."""
    steps = float(seconds) / float(step_dt)
    if exact and (abs(steps - round(steps)) > 1e-6 or round(steps) < 1):
        raise ValueError(f"{name} = {seconds} s is not a whole number of {step_dt:g} s steps ({steps:g}).")
    return max(1, int(round(steps)))


class BaseModule(ABC):
    """Minimal lifecycle interface for a modular environment component.

    Subclasses override only the hooks they need.  The manager dispatches each
    hook to every registered module in the order given by the YAML ``modules``
    list.
    """

    # Training curriculum progress (rl_games 'curriculum' levels, e.g. 0 -> 1); 1.0 = the full task (play, export).
    curriculum_factor: float = 1.0
    # True: the module must run every physics step (pre_gym / post_gym phases) when a control step has several
    # (frame_skip > 1), e.g. a servo law that integrates at the physics step; checked by ModularEnvironment.
    physics_rate: bool = False

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

    def set_curriculum_factor(self, factor: float, container: ModuleContainer) -> None:
        """Called whenever the training curriculum changes level; modules read ``self.curriculum_factor``."""
        self.curriculum_factor = float(factor)

    def reset(self, container: ModuleContainer) -> None:
        """Reset the modules selected by the environment's reset buffer.

        This maps to the existing ``reset_idx`` convention.
        """
        pass

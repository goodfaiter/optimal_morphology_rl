"""Module that assigns RGB materials to articulation links by name."""

from __future__ import annotations

from typing import Any

from optimal_morphology_rl.modules.articulation_link_colorer import (
    ArticulationLinkColorer,
)
from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


# Default hand coloring used across all hand-object tasks.
_DEFAULT_HAND_COLOR_MAP = {
    "palm": (0.95, 0.82, 0.72),      # skin tone
    "thumb_0": (0.95, 0.60, 0.20),   # orange
    "thumb_1": (0.75, 0.35, 0.90),   # purple
    "finger_0": (0.95, 0.30, 0.30),  # red
    "finger_1": (0.30, 0.70, 0.40),  # green
    "finger_2": (0.25, 0.55, 0.95),  # blue
    "finger_3": (1.00, 0.85, 0.20),  # yellow
    "finger_4": (0.95, 0.50, 0.70),  # pink
}


@register_module("articulation_link_colorer")
class ArticulationLinkColorerModule(BaseModule):
    """Colors articulation links based on their names.

    The module expects ``container.robot`` to be populated by the ``create_robot``
    module.  It runs in ``finalize`` (after the robot articulation has been
    loaded into the environment definition) and assigns RGB materials to
    matched links.

    Config shape::

        articulation_link_colorer:
          color_map:
            palm: [0.95, 0.82, 0.72]
            finger_0: [0.95, 0.30, 0.30]
          exclude_substrings:
            - _abd
            - _base
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        color_map = dict(
            self.config.get("color_map", _DEFAULT_HAND_COLOR_MAP)
        )
        exclude_substrings = tuple(
            self.config.get("exclude_substrings", ("_abd", "_base"))
        )
        self.colorer = ArticulationLinkColorer(color_map, exclude_substrings)

    def finalize(self, container: ModuleContainer) -> None:
        """Assign colors to the robot articulation links."""
        robot = container.get("robot")
        if robot is None:
            raise RuntimeError(
                "ArticulationLinkColorerModule requires 'robot' in the shared container. "
                "Ensure the 'create_robot' module is listed before 'articulation_link_colorer'."
            )
        if container.get("env_def") is None:
            raise RuntimeError(
                "ArticulationLinkColorerModule requires 'env_def' in the shared container."
            )

        self.colorer.assign(container.env_def, robot.def_handle, robot.art_def)

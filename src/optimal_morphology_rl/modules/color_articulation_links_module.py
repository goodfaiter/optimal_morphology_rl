"""Module that assigns RGB materials to articulation links by name."""

from __future__ import annotations

from typing import Any

import vlearn as v

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


class ColorArticulationLinks:
    """Assign RGB materials to articulation links based on link-name matching."""

    def __init__(
        self,
        color_map: dict[str, tuple[float, float, float]],
        exclude_substrings: tuple[str, ...] = ("_abd", "_base"),
    ):
        self.color_map = color_map
        self.exclude_substrings = exclude_substrings

    def assign(self, env_def, arti_def_handle, art_def) -> None:
        """Create RGB materials and assign one to each matched link."""
        link_colors: list[tuple[int, tuple[float, float, float]]] = []
        for i in range(art_def.get_num_link_defs()):
            link_name = art_def.get_link_def(i).name
            color = self._color_for_link(link_name)
            if color is not None:
                link_colors.append((i, color))

        if not link_colors:
            return

        used_colors = {color for _, color in link_colors}
        color_to_handle: dict[tuple[float, float, float], int] = {}
        for color in used_colors:
            rgb_mat = v.RGBMaterial()
            rgb_mat.color = v.Vec3(*color)
            rgb_mat.specular = 40.0
            rgb_mat.spec_intensity = 0.25
            color_to_handle[color] = env_def.create_rgb_material(rgb_mat)

        for i, color in link_colors:
            env_def.assign_rgb_material_to_articulation_link(
                arti_def_handle, color_to_handle[color], i
            )

    def _color_for_link(self, link_name: str) -> tuple[float, float, float] | None:
        """Return the color for a link, or None if the link should be skipped."""
        name_lower = link_name.lower()
        if any(excl in name_lower for excl in self.exclude_substrings):
            return None
        for key, color in self.color_map.items():
            if key.lower() in name_lower:
                return color
        return None


@register_module("color_articulation_links")
class ColorArticulationLinksModule(BaseModule):
    """Colors articulation links based on their names.

    The module expects ``container.robot`` to be populated by the ``create_robot``
    module.  It runs in ``finalize`` (after the robot articulation has been
    loaded into the environment definition) and assigns RGB materials to
    matched links.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        color_map = dict(
            self.config.get("color_map", _DEFAULT_HAND_COLOR_MAP)
        )
        exclude_substrings = tuple(
            self.config.get("exclude_substrings", ("_abd", "_base"))
        )
        self.colorer = ColorArticulationLinks(color_map, exclude_substrings)

    def finalize(self, container: ModuleContainer) -> None:
        """Assign colors to the robot articulation links."""
        robot = container.get("robot")
        if robot is None:
            raise RuntimeError(
                "ColorArticulationLinksModule requires 'robot' in the shared container. "
                "Ensure the 'create_robot' module is listed before 'color_articulation_links'."
            )
        if container.get("env_def") is None:
            raise RuntimeError(
                "ColorArticulationLinksModule requires 'env_def' in the shared container."
            )

        self.colorer.assign(container.env_def, robot.def_handle, robot.art_def)

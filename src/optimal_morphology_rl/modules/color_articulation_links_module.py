"""Module that assigns RGB materials to articulation links (by name group) and to the loaded objects."""

from __future__ import annotations

import re
from typing import Any

import vlearn as v

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.create_objects_module import LoadedRigidObject
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module

# Qualitative palette (matplotlib tab10), assigned in order to the link groups, then to the objects.
_PALETTE = (
    (0.122, 0.467, 0.706),  # blue
    (1.000, 0.498, 0.055),  # orange
    (0.173, 0.627, 0.173),  # green
    (0.839, 0.153, 0.157),  # red
    (0.580, 0.404, 0.741),  # purple
    (0.549, 0.337, 0.294),  # brown
    (0.890, 0.467, 0.761),  # pink
    (0.498, 0.498, 0.498),  # gray
    (0.737, 0.741, 0.133),  # olive
    (0.090, 0.745, 0.812),  # cyan
)

# A link's group: its leading letters plus an optional index ('Index_MCP_jointbody' -> 'index', 'finger_0_link' ->
# 'finger_0'), so all links of one digit share a color.
_GROUP_PATTERN = re.compile(r"^[A-Za-z]+(?:_\d+)?")


def _rgb_material(env_def, color: tuple[float, float, float]):
    rgb_mat = v.RGBMaterial()
    rgb_mat.color = v.Vec3(*color)
    rgb_mat.specular = 40.0
    rgb_mat.spec_intensity = 0.25
    return env_def.create_rgb_material(rgb_mat)


def link_group(link_name: str) -> str:
    """The color group of a link name (see _GROUP_PATTERN)."""
    match = _GROUP_PATTERN.match(link_name)
    return (match.group(0) if match else link_name).lower()


class ColorArticulationLinks:
    """Assign RGB materials to articulation links: one palette color per link group (or an explicit color map)."""

    def __init__(
        self,
        color_map: dict[str, tuple[float, float, float]] | None = None,
        exclude_substrings: tuple[str, ...] = ("_abd", "_base"),
    ):
        self.color_map = color_map
        self.exclude_substrings = exclude_substrings
        self.group_colors: dict[str, tuple[float, float, float]] = {}

    def assign(self, env_def, arti_def_handle, art_def) -> None:
        """Create RGB materials and assign one to each colored link."""
        link_colors: list[tuple[int, tuple[float, float, float]]] = []
        for i in range(art_def.get_num_link_defs()):
            color = self._color_for_link(art_def.get_link_def(i).name)
            if color is not None:
                link_colors.append((i, color))

        color_to_handle = {color: _rgb_material(env_def, color) for color in {color for _, color in link_colors}}
        for i, color in link_colors:
            env_def.assign_rgb_material_to_articulation_link(arti_def_handle, color_to_handle[color], i)

    def next_color(self) -> tuple[float, float, float]:
        """The next unused palette color (after the link groups), for objects."""
        color = _PALETTE[len(self.group_colors) % len(_PALETTE)]
        self.group_colors[f"__object_{len(self.group_colors)}"] = color
        return color

    def _color_for_link(self, link_name: str) -> tuple[float, float, float] | None:
        """Return the color for a link, or None if the link should be skipped."""
        name_lower = link_name.lower()
        if any(excl.lower() in name_lower for excl in self.exclude_substrings):
            return None
        if self.color_map is not None:
            for key, color in self.color_map.items():
                if key.lower() in name_lower:
                    return color
            return None
        group = link_group(link_name)
        if group not in self.group_colors:
            self.group_colors[group] = _PALETTE[len(self.group_colors) % len(_PALETTE)]
        return self.group_colors[group]


@register_module("color_articulation_links")
class ColorArticulationLinksModule(BaseModule):
    """Colors the robot's articulation links and the loaded (non-fixed) rigid objects for rendering.

    Links are grouped by name (leading letters plus an optional index: 'Index_MCP_jointbody' -> index,
    'finger_0_link' -> finger_0) and each group gets the next color of a qualitative palette (tab10), so every digit
    has its own color; links matching ``exclude_substrings`` keep their material. Each non-fixed rigid object of
    ``create_objects`` (listed before this module) then gets the next palette color; fixed objects (e.g. a table)
    keep theirs. ``color_map`` (link-name substring -> RGB) / ``object_colors`` (object name -> RGB) override the
    generated colors.

    The module expects ``container.robot`` (``create_robot``) and runs in ``finalize``, after the robot and the objects
    are loaded into the environment definition.

    Config::

        color_articulation_links:
          exclude_substrings: [_abd, _base, slider]   # default [_abd, _base]
          color_map: {palm: [0.95, 0.82, 0.72]}       # optional
          object_colors: {cube: [0.0, 0.75, 0.75]}    # optional
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        color_map = self.config.get("color_map")
        if color_map is not None:
            color_map = {key: tuple(color) for key, color in color_map.items()}
        exclude_substrings = tuple(self.config.get("exclude_substrings", ("_abd", "_base")))
        self.colorer = ColorArticulationLinks(color_map, exclude_substrings)
        self.object_colors = {name: tuple(color) for name, color in (self.config.get("object_colors") or {}).items()}

    def finalize(self, container: ModuleContainer) -> None:
        """Assign colors to the robot articulation links and the objects."""
        robot = container.get("robot")
        if robot is None:
            raise RuntimeError(
                "ColorArticulationLinksModule requires 'robot' in the shared container. "
                "Ensure the 'create_robot' module is listed before 'color_articulation_links'."
            )
        if container.get("env_def") is None:
            raise RuntimeError("ColorArticulationLinksModule requires 'env_def' in the shared container.")

        self.colorer.assign(container.env_def, robot.def_handle, robot.art_def)

        objects = container.get("objects") or {}
        unknown = [name for name in self.object_colors if name not in objects]
        if unknown:
            raise RuntimeError(f"ColorArticulationLinksModule: objects {unknown} are not created (list create_objects first).")
        for name, obj in objects.items():
            if not isinstance(obj, LoadedRigidObject) or obj.fixed:
                continue
            color = self.object_colors.get(name) or self.colorer.next_color()
            handle = container.env_def.get_rigid_body_def_handle_by_name(obj.name)
            container.env_def.assign_rgb_material_to_rigid_body(handle, _rgb_material(container.env_def, color))

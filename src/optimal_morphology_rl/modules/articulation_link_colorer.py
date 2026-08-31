import vlearn as v


class ArticulationLinkColorer:
    """Assign RGB materials to articulation links based on link-name matching.

    This is useful when the underlying asset does not embed its own visual
    materials (or when we want to override them).  It requires the articulation
    to be loaded with ``use_visual_mesh=True`` so that each link has visual
    geometry for the material to attach to.
    """

    def __init__(
        self,
        color_map: dict[str, tuple[float, float, float]],
        exclude_substrings: tuple[str, ...] = ("_abd", "_base"),
    ):
        """
        Args:
            color_map: Mapping from name substring to RGB color.  A link whose
                lowercase name contains a key receives that key's color.
                Links that do not match any key are left unchanged.
            exclude_substrings: Link names containing any of these substrings
                are skipped.  Useful for virtual/no-visual links such as
                ``*_abd`` (abduction intermediate links) or ``*_base``.
        """
        self.color_map = color_map
        self.exclude_substrings = exclude_substrings

    def assign(self, env_def, arti_def_handle, art_def) -> None:
        """Create RGB materials and assign one to each matched link."""
        # First pass: determine which links match and collect the colors used.
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

        # Second pass: assign materials only to matched links.
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

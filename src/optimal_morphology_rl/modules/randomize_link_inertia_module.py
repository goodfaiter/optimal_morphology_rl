"""Module that randomizes the robot link masses and diagonal inertias."""

from __future__ import annotations

from typing import Any

import torch
import vlearn as v

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


def _scale_range(config: dict[str, Any], key: str) -> tuple[float, float]:
    """Validate a positive ``[min, max]`` scale range (default: no scaling)."""
    value = config.get(key, [1.0, 1.0])
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise RuntimeError(f"RandomizeLinkInertiaModule config '{key}' must be a [min, max] list, got {value}.")
    low, high = float(value[0]), float(value[1])
    if not 0.0 < low <= high:
        raise RuntimeError(f"RandomizeLinkInertiaModule config '{key}' must satisfy 0 < min <= max, got {value}.")
    return low, high


@register_module("randomize_link_inertia")
class RandomizeLinkInertiaModule(BaseModule):
    """Randomizes the mass and diagonal inertia of robot links on reset.

    Per link, a mass scale ``s_m`` from ``mass_scale_range`` and an inertia scale ``s_i`` from
    ``inertia_scale_range`` are drawn uniformly, and the link gets

        mass    = s_m * defined_mass
        inertia = s_m * s_i * defined_diag_inertia      (all three axes)

    i.e. the inertia follows the mass (same shape, different density) and ``s_i`` adds shape
    uncertainty on top. Link properties belong to the shared articulation definition, so there is
    one value per env set (per env with ``create_rigid_vsim_envs.num_envs_per_set: 1``); an env
    set is resampled when any of its envs resets.

    vlearn rescales a link's inertia by the same factor when its mass is written, so the inertia
    is always written explicitly after the mass. ``robot.link_masses`` becomes
    ``(num_envs, num_links)`` and tracks the sampled masses (gravity compensation reads it).

    ``links`` (optional) selects the links by name; by default every link except the
    'create_spool' tendon sliders, whose mass is part of the spool inertia.

    Config shape::

        randomize_link_inertia:
          mass_scale_range: [0.8, 1.2]
          inertia_scale_range: [0.8, 1.2]
          links: [Pinky_MCP_jointbody, Pinky_PIP_jointbody, Pinky_DP]  # optional
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.mass_range = _scale_range(self.config, "mass_scale_range")
        self.inertia_range = _scale_range(self.config, "inertia_scale_range")
        self.link_names = self.config.get("links")

    def finalize(self, container: ModuleContainer) -> None:
        if container.get("robot") is None:
            raise RuntimeError(
                "RandomizeLinkInertiaModule requires 'robot' in the shared container. "
                "Ensure the 'create_robot' module is listed before 'randomize_link_inertia'."
            )

    def post_finalize(self, container: ModuleContainer) -> None:
        robot = container.robot
        device = container.device
        num_links = robot.num_links
        num_env_sets = len(container.num_envs)
        names = [robot.art_def.get_link_def(i).name for i in range(num_links)]

        spool = container.get("spool")
        sliders = set() if spool is None else set(spool.slider_link_names)
        if self.link_names is None:
            selected = [name for name in names if name not in sliders]
        else:
            selected = list(self.link_names)
            unknown = [name for name in selected if name not in names]
            if unknown:
                raise RuntimeError(f"RandomizeLinkInertiaModule 'links' {unknown} are not robot links {names}.")
            on_sliders = [name for name in selected if name in sliders]
            if on_sliders:
                raise RuntimeError(
                    f"RandomizeLinkInertiaModule: {on_sliders} are 'create_spool' sliders (their mass is part of the spool inertia)."
                )
        self.link_mask = torch.tensor([name in selected for name in names], device=device)
        self.selected = selected

        defs = [robot.art_def.get_link_def(i) for i in range(num_links)]
        self.defined_mass = torch.tensor([d.mass for d in defs], device=device, dtype=torch.float32)
        self.defined_inertia = torch.tensor(
            [[d.diag_inertia.x, d.diag_inertia.y, d.diag_inertia.z] for d in defs], device=device, dtype=torch.float32
        )
        # Env set of every env (envs are numbered set by set).
        self.env_set_index = torch.repeat_interleave(
            torch.arange(num_env_sets, device=device), torch.tensor(container.num_envs, device=device)
        )

        # The commands write every link, so the other links keep their defined values.
        container.set_link_mass_buf = self.defined_mass.repeat(num_env_sets, 1)
        container.set_link_inertia_buf = self.defined_inertia.reshape(-1).repeat(num_env_sets, 1)
        self.mass_scale = torch.ones((num_env_sets, num_links), device=device)
        self.inertia_scale = torch.ones((num_env_sets, num_links), device=device)

        def command(prop, buf):
            cmd = container.env_group.create_link_property_command(prop, v.wrap_gpu_buffer(buf), robot.def_handle, (0, num_links))
            return container.gym.create_gpu_array([cmd])

        container.gpu_set_link_mass_cmd = command(v.LinkProperty.MASS, container.set_link_mass_buf)
        container.gpu_set_link_inertia_cmd = command(v.LinkProperty.DIAG_INERTIA, container.set_link_inertia_buf)
        robot.link_masses = self.defined_mass.expand(container.total_num_envs, num_links).clone()
        print(
            f"RandomizeLinkInertiaModule: links {selected}, mass scale {list(self.mass_range)}, "
            f"inertia scale {list(self.inertia_range)}"
        )

    def reset(self, container: ModuleContainer) -> None:
        """Resample the env sets that contain a resetting env and write mass, then inertia."""
        reset_buf = container.reset_buf
        if not reset_buf.any():
            return
        num_env_sets = self.mass_scale.shape[0]
        set_mask = torch.zeros(num_env_sets, dtype=torch.bool, device=reset_buf.device)
        set_mask[self.env_set_index[reset_buf]] = True
        rows = set_mask.unsqueeze(1) & self.link_mask

        def draw(low_high):
            low, high = low_high
            return low + (high - low) * torch.rand(self.mass_scale.shape, device=self.mass_scale.device)

        self.mass_scale[:] = torch.where(rows, draw(self.mass_range), self.mass_scale)
        self.inertia_scale[:] = torch.where(rows, draw(self.inertia_range), self.inertia_scale)

        container.set_link_mass_buf[:] = self.mass_scale * self.defined_mass
        inertia = (self.mass_scale * self.inertia_scale).unsqueeze(-1) * self.defined_inertia
        container.set_link_inertia_buf[:] = inertia.reshape(num_env_sets, -1)
        # Mass first: vlearn rescales the inertia with it, so the inertia write must come after.
        container.gym.set_link_properties(container.gpu_set_link_mass_cmd)
        container.gym.set_link_properties(container.gpu_set_link_inertia_cmd)
        container.robot.link_masses[:] = container.set_link_mass_buf[self.env_set_index]

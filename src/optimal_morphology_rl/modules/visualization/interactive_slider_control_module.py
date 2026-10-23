"""Interactive slider control for visual testing of robot hands."""

from __future__ import annotations

from typing import Any

import torch
import vlearn as v

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("interactive_slider_control")
class InteractiveSliderControlModule(BaseModule):
    """Creates UI sliders that drive the robot's action vector directly.

    This module is intended for visual testing only. During
    ``pre_physics_step`` it reads slider values from the vlearn renderer and
    writes them into ``container.actions``. Downstream ``process_actions`` and
    ``robot_control`` modules then treat the slider values as the policy
    output, so the user is effectively sending action commands to the hand.

    One slider is created per action, regardless of tendons/motors, with the
    range given by the required ``slider_min`` / ``slider_max`` config keys;
    with ``shared: true`` a single slider drives every action (e.g. curl all
    fingers together). Resetting is done with the render module's "Reset"
    checkbox.

    With ``spool_positions: true`` there is one slider per spool ('create_spool') and its value is the desired spool
    angle (rad); every step the action becomes the change to it, target - angle (unscaled through
    'process_variable_actions'), as the delta action the policy outputs.

    Config shape::

        interactive_slider_control:
          slider_min: 0.0        # required slider lower bound
          slider_max: 10.0       # required slider upper bound
          shared: false          # optional: one slider for all actions
          spool_positions: false # optional: sliders are desired spool angles (rad)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.enabled = bool(self.config.get("enabled", True))
        self.slider_min = self.config.get("slider_min")
        self.slider_max = self.config.get("slider_max")
        self.shared = bool(self.config.get("shared", False))
        self.spool_positions = bool(self.config.get("spool_positions", False))

        self.sliders: list[v.UserSlider] = []
        self._num_actions: int | None = None

    def finalize(self, container: ModuleContainer) -> None:
        """Validate that the robot is available and the slider range is configured.

        ``num_actions`` is set by ``robot_control`` during its own finalize; it
        is read lazily in ``post_finalize`` so this module can appear before
        ``robot_control`` in the config while still creating UI widgets that
        depend on the action-space size.
        """
        if container.get("robot") is None:
            raise RuntimeError(
                "InteractiveSliderControlModule requires 'robot' in the shared container. "
                "Ensure 'create_robot' is listed before this module."
            )
        if self.enabled:
            missing = [name for name in ("slider_min", "slider_max") if self.config.get(name) is None]
            if missing:
                raise RuntimeError(f"InteractiveSliderControlModule config missing {missing}: the slider range.")

    def post_finalize(self, container: ModuleContainer) -> None:
        """Create renderer UI widgets once the renderer is available."""
        self.sliders.clear()

        if not self.enabled:
            return

        env = container.get("env")
        if env is None or not getattr(env, "rendering", False):
            return

        gym_render = container.get("gym_render", None)
        if gym_render is None:
            return

        num_actions = container.get("num_actions", None)
        if num_actions is None:
            return
        self._num_actions = num_actions

        if self.spool_positions:
            spool = container.get("spool")
            if spool is None:
                raise RuntimeError("InteractiveSliderControlModule 'spool_positions' requires 'create_spool'.")
            names = [name.removesuffix("_slider") for name in spool.slider_link_names]
            labels = ["All spools [rad]"] if self.shared else [f"{name} [rad]" for name in names]
        else:
            labels = ["All actions"] if self.shared else [f"Action {i}" for i in range(num_actions)]
        for label in labels:
            slider = v.UserSlider(label, self.slider_min, self.slider_max, 0.0)
            gym_render.register_menu_item(slider)
            self.sliders.append(slider)

    def step(self, container: ModuleContainer) -> None:
        """Read slider values and write them to ``container.actions``."""
        if not self.enabled or not self.sliders:
            return

        if self._num_actions is None:
            return

        values = [slider.get_value() for slider in self.sliders]
        if self.spool_positions:
            self._write_spool_deltas(container, values)
            return
        if self.shared:
            values = values * self._num_actions
        actions = torch.tensor(
            values,
            dtype=torch.float32,
            device=container.device,
        )
        container.actions[:] = actions.unsqueeze(0).expand(
            container.total_num_envs, -1
        )

    def _write_spool_deltas(self, container: ModuleContainer, values: list[float]) -> None:
        """Actions that move each spool to its slider angle: delta = target - theta_m, before the action scaling."""
        spool = container.spool
        target = torch.tensor(values, dtype=torch.float32, device=container.device)
        # The current spool angle, as 'create_spool' reads it this step (its theta_m is still the last step's).
        theta = container.robot.get_joint_pos_buf[:, spool.dof_indices] / spool.pully_radius
        delta = target - theta  # broadcasts a shared slider
        lo, hi = container.dof_scale_range
        columns = torch.arange(container.actions.shape[1], device=container.device)[container.active_dof_slice][spool.action_slots]
        lo, hi = lo[spool.action_slots], hi[spool.action_slots]
        container.actions[:, columns] = 2.0 * (delta - lo) / (hi - lo) - 1.0  # inverse of vlearn's scale()

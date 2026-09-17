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
    range given by the required ``slider_min`` / ``slider_max`` config keys. A
    reset checkbox is also exposed (label ``reset_name``, default
    ``"Reset"``); the runner polls it and calls ``env.reset()`` when it
    becomes checked.

    Config shape::

        interactive_slider_control:
          slider_min: 0.0        # required slider lower bound
          slider_max: 10.0       # required slider upper bound
          reset_name: "Reset"    # optional reset checkbox label
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.enabled = bool(self.config.get("enabled", True))
        self.reset_name = str(self.config.get("reset_name", "Reset"))
        self.slider_min = self.config.get("slider_min")
        self.slider_max = self.config.get("slider_max")

        self.sliders: list[v.UserSlider] = []
        self.reset_checkbox: v.UserCheckbox | None = None
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
        self.reset_checkbox = None

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

        self.reset_checkbox = v.UserCheckbox(self.reset_name, False)
        gym_render.register_menu_item(self.reset_checkbox)

        for i in range(num_actions):
            slider = v.UserSlider(f"Action {i}", self.slider_min, self.slider_max, 0.0)
            gym_render.register_menu_item(slider)
            self.sliders.append(slider)



    def step(self, container: ModuleContainer) -> None:
        """Read slider values and write them to ``container.actions``."""
        if not self.enabled or not self.sliders:
            return

        if self._num_actions is None:
            return

        values = [slider.get_value() for slider in self.sliders]
        actions = torch.tensor(
            values,
            dtype=torch.float32,
            device=container.device,
        )
        container.actions[:] = actions.unsqueeze(0).expand(
            container.total_num_envs, -1
        )

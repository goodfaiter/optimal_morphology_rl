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
    output, so the user is effectively sending normalized motor-force / root-
    velocity commands to the hand.

    The slider labels and ranges depend on ``robot.use_tendon``:

    - Tendon-driven hands: one slider per spatial tendon, range ``[0, 10]``.
    - Motor-driven hands: one slider per active motor, range ``[-1, 1]``.

    A reset checkbox is also exposed; the runner is expected to poll it and
    call ``env.reset()`` when it becomes checked.

    Config shape::

        interactive_slider_control:
          enabled: true          # set to false to disable the sliders
          reset_name: "Reset"    # label for the reset checkbox
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.enabled = bool(self.config.get("enabled", True))
        self.reset_name = str(self.config.get("reset_name", "Reset"))

        self.sliders: list[v.UserSlider] = []
        self.reset_checkbox: v.UserCheckbox | None = None
        self._num_actions: int | None = None

    def finalize(self, container: ModuleContainer) -> None:
        """Validate that the robot is available.

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

        robot = container.robot
        num_actions = container.get("num_actions", None)
        if num_actions is None:
            return
        self._num_actions = num_actions

        self.reset_checkbox = v.UserCheckbox(self.reset_name, False)
        gym_render.register_menu_item(self.reset_checkbox)

        active_indices = container.get("active_motor_indices", None)
        is_floating = not robot.fixed_hand

        root_names = [
            "Root AngVel X",
            "Root AngVel Y",
            "Root AngVel Z",
            "Root LinVel X",
            "Root LinVel Y",
            "Root LinVel Z",
        ]

        for i in range(num_actions):
            if is_floating and i < 6:
                name = root_names[i]
                slider_min, slider_max = -1.0, 1.0
            else:
                dof_slot = i if robot.fixed_hand else i - 6
                if active_indices is not None and dof_slot < len(active_indices):
                    real_idx = int(active_indices[dof_slot].item())
                    if robot.use_tendon:
                        tendon_def = robot.art_def.get_spatial_tendon_def(real_idx)
                        name = tendon_def.name if tendon_def.name else f"Tendon {real_idx}"
                        slider_min, slider_max = 0.0, 10.0
                    else:
                        motor_def = robot.art_def.get_motor_def(real_idx)
                        name = motor_def.name if motor_def.name else f"Motor {real_idx}"
                        slider_min, slider_max = -1.0, 1.0
                else:
                    name = f"Action {i}"
                    slider_min, slider_max = -1.0, 1.0

            slider = v.UserSlider(name, slider_min, slider_max, 0.0)
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

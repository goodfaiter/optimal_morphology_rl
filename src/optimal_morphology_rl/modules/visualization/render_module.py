"""Module that owns all rendering timing, camera, and window setup."""

from __future__ import annotations

from typing import Any

import vlearn as v

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("render")
class RenderModule(BaseModule):
    """Renders the environment at the right points in the simulation loop.

    This module is intentionally separate from :class:`ModularEnvironment` so
    that rendering behavior can be configured per-task or omitted entirely.

    Config shape::

        render:
          capped_step: false     # true = user must advance each frame
          paused: false          # start with renderer paused
          raise_exception: null  # null -> same as env.rendering
          camera:
            eye: [-0.671139, 0.073098, 0.726423]
            target: [0.755459, -0.009100, -0.655133]
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.camera = dict(
            self.config.get(
                "camera",
                {
                    "eye": [-0.671139, 0.073098, 0.726423],
                    "target": [0.755459, -0.009100, -0.655133],
                },
            )
        )

    def post_finalize(self, container: ModuleContainer) -> None:
        """Obtain the renderer and configure camera/window behavior."""
        env = container.env
        if not getattr(env, "rendering", False):
            return

        # The render handle is created here rather than in create_rigid_vsim_envs
        # so that all rendering setup lives in one module.
        gym_render = env.gym.get_render()
        if gym_render is None:
            return

        container.gym_render = gym_render

        eye = self.camera.get("eye", [-0.671139, 0.073098, 0.726423])
        target = self.camera.get("target", [0.755459, -0.009100, -0.655133])
        gym_render.reset_camera(v.Vec3(*eye), v.Vec3(*target))
        gym_render.capped_step = False
        gym_render.set_paused(False)

    def step(self, container: ModuleContainer) -> None:
        """Render the environment and mark the simulation step as finished."""
        env = container.env
        gym_render = container.get("gym_render", None)
        if not getattr(env, "rendering", False) or gym_render is None:
            return

        finished = gym_render.render(lambda: None)
        if finished and self.raise_exception:
            raise RuntimeError("Render window was closed.")
        gym_render.set_step(False)

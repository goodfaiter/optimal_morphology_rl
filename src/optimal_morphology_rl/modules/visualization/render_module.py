"""Module that owns all rendering timing, camera, and window setup."""

from __future__ import annotations

from typing import Any

import vlearn as v

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("render")
class RenderModule(BaseModule):
    """Renders the environment at the right points in the simulation loop.

    This module is intentionally separate from :class:`ModularEnvironment` so
    that rendering behavior can be configured per-task or omitted entirely.

    Config shape::

        render:
          render_substep: true   # render inside each gym sub-step
          capped_step: false     # true = user must advance each frame
          paused: false          # start with renderer paused
          raise_exception: null  # null -> same as env.rendering
          camera:
            eye: [-0.671139, 0.073098, 0.726423]
            target: [0.755459, -0.009100, -0.655133]
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.render_substep = bool(self.config.get("render_substep", True))
        self.capped_step = bool(self.config.get("capped_step", False))
        self.paused = bool(self.config.get("paused", False))
        self.raise_exception = self.config.get("raise_exception", None)
        self.camera = dict(
            self.config.get(
                "camera",
                {
                    "eye": [-0.671139, 0.073098, 0.726423],
                    "target": [0.755459, -0.009100, -0.655133],
                },
            )
        )

    def finalize(self, env: Any) -> None:
        """Store the exception-on-close flag on the environment."""
        if self.raise_exception is None:
            env.raise_exception = getattr(env, "rendering", False)
        else:
            env.raise_exception = bool(self.raise_exception)

    def post_finalize(self, env: Any) -> None:
        """Obtain the renderer and configure camera/window behavior."""
        if not getattr(env, "rendering", False):
            return

        # The render handle is created here rather than in create_rigid_vsim_envs
        # so that all rendering setup lives in one module.
        gym_render = env.gym.get_render()
        if gym_render is None:
            return

        env.module_manager.container.gym_render = gym_render

        eye = self.camera.get("eye", [-0.671139, 0.073098, 0.726423])
        target = self.camera.get("target", [0.755459, -0.009100, -0.655133])
        gym_render.reset_camera(v.Vec3(*eye), v.Vec3(*target))
        gym_render.capped_step = self.capped_step
        gym_render.set_paused(self.paused)

    def pre_physics_step(self, env: Any) -> None:
        """Render once per control step when not rendering every sub-step."""
        if getattr(env, "rendering", False) and not self.render_substep:
            env.render()

    def pre_gym_step(self, env: Any) -> None:
        """Render before each gym sub-step when sub-step rendering is enabled."""
        if getattr(env, "rendering", False) and self.render_substep:
            env.render()

    def post_physics_step(self, env: Any) -> None:
        """Tell the renderer that the simulation step is finished."""
        if getattr(env, "rendering", False) and env.gym_render is not None:
            env.gym_render.set_step(False)

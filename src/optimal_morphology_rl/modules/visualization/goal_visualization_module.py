"""Module that draws the reward object's goal axes in the renderer."""

from __future__ import annotations

from typing import Any

import torch
import vlearn as v
from vlearn.torch_utils.torch_jit_utils import quat_rotate

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("goal_visualization")
class GoalVisualizationModule(BaseModule):
    """Draws RGB axes at the reward object's current goal pose.

    This module only does work when rendering is enabled.  It updates the
    debug line shapes during ``post_physics_step`` so the renderer can draw
    them on the next frame.

    Config shape::

        goal_visualization:
          line_width: 3.0
          axis_length: 0.1
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.line_width = float(self.config.get("line_width", 3.0))
        self.axis_length = float(self.config.get("axis_length", 0.1))

    def post_physics_step(self, env: Any) -> None:
        """Update goal-axis visualization for the first environment."""
        if not getattr(env, "rendering", False) or env.gym_render is None:
            return

        container = env.module_manager.container
        reward_object = container.get("reward_object")
        if reward_object is None:
            return

        goal_pos = reward_object.goal_pos_in_world[0]
        goal_quat = reward_object.goal_quat_object_to_world[0:1]

        axes = [
            quat_rotate(
                goal_quat,
                torch.tensor(
                    [[1.0, 0.0, 0.0]], device=env.device, dtype=torch.float32
                ),
            )[0],
            quat_rotate(
                goal_quat,
                torch.tensor(
                    [[0.0, 1.0, 0.0]], device=env.device, dtype=torch.float32
                ),
            )[0],
            quat_rotate(
                goal_quat,
                torch.tensor(
                    [[0.0, 0.0, 1.0]], device=env.device, dtype=torch.float32
                ),
            )[0],
        ]

        goal_points = [
            [
                v.Vec3(goal_pos[0].item(), goal_pos[1].item(), goal_pos[2].item()),
                v.Vec3(
                    (goal_pos[0] + self.axis_length * axes[0][0]).item(),
                    (goal_pos[1] + self.axis_length * axes[0][1]).item(),
                    (goal_pos[2] + self.axis_length * axes[0][2]).item(),
                ),
            ],
            [
                v.Vec3(goal_pos[0].item(), goal_pos[1].item(), goal_pos[2].item()),
                v.Vec3(
                    (goal_pos[0] + self.axis_length * axes[1][0]).item(),
                    (goal_pos[1] + self.axis_length * axes[1][1]).item(),
                    (goal_pos[2] + self.axis_length * axes[1][2]).item(),
                ),
            ],
            [
                v.Vec3(goal_pos[0].item(), goal_pos[1].item(), goal_pos[2].item()),
                v.Vec3(
                    (goal_pos[0] + self.axis_length * axes[2][0]).item(),
                    (goal_pos[1] + self.axis_length * axes[2][1]).item(),
                    (goal_pos[2] + self.axis_length * axes[2][2]).item(),
                ),
            ],
        ]

        for attr_name in ("_goal_axis_x", "_goal_axis_y", "_goal_axis_z"):
            line = getattr(self, attr_name, None)
            if line is not None:
                env.gym_render.unregister_line_shape(line)

        env_set = env.env_sets[0]
        env_handle = env_set.get_environment_handle(0)

        colors = [
            v.Vec3(1.0, 0.0, 0.0),
            v.Vec3(0.0, 1.0, 0.0),
            v.Vec3(0.0, 0.0, 1.0),
        ]
        attr_names = ("_goal_axis_x", "_goal_axis_y", "_goal_axis_z")

        for attr_name, color, points in zip(attr_names, colors, goal_points):
            line = env.gym_render.create_user_line(
                points,
                color,
                line_width=self.line_width,
                visible=True,
                env_handle=env_handle,
            )
            env.gym_render.register_line_shape(line)
            setattr(self, attr_name, line)

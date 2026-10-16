"""Module that draws the reward object's goal axes in the renderer."""

from __future__ import annotations

from typing import Any

import torch
import vlearn as v
from vlearn.torch_utils.torch_jit_utils import quat_rotate

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.goal_pose_module import goal_in_env_frame
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("goal_visualization")
class GoalVisualizationModule(BaseModule):
    """Draws RGB axes at the goal pose stored in the shared container.

    The goal is the ``goal_pose`` module's goal, converted to the environment frame
    (:func:`~optimal_morphology_rl.modules.goal_pose_module.goal_in_env_frame`).

    This module only does work when rendering is enabled and the goal exists.
    It updates the debug line shapes during the post-gym step so the renderer
    can draw them on the next frame.

    Config shape::

        goal_visualization:
          line_width: 3.0
          axis_length: 0.025
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.line_width = float(self.config.get("line_width", 3.0))
        self.axis_length = float(self.config.get("axis_length", 0.025))

    def step(self, container: ModuleContainer) -> None:
        """Update goal-axis visualization for the first environment."""
        env = container.env
        if not getattr(env, "rendering", False) or container.gym_render is None:
            return

        if container.get("goal_position_in_goal_frame") is None:
            return
        pos_goal_in_world, quat_goal_to_world = goal_in_env_frame(container)

        goal_pos = pos_goal_in_world[0]
        goal_quat = quat_goal_to_world[0:1]

        axes = [
            quat_rotate(
                goal_quat,
                torch.tensor([[1.0, 0.0, 0.0]], device=env.device, dtype=torch.float32),
            )[0],
            quat_rotate(
                goal_quat,
                torch.tensor([[0.0, 1.0, 0.0]], device=env.device, dtype=torch.float32),
            )[0],
            quat_rotate(
                goal_quat,
                torch.tensor([[0.0, 0.0, 1.0]], device=env.device, dtype=torch.float32),
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
                container.gym_render.unregister_line_shape(line)

        env_set = container.env_sets[0]
        env_handle = env_set.get_environment_handle(0)

        colors = [
            v.Vec3(1.0, 0.0, 0.0),
            v.Vec3(0.0, 1.0, 0.0),
            v.Vec3(0.0, 0.0, 1.0),
        ]
        attr_names = ("_goal_axis_x", "_goal_axis_y", "_goal_axis_z")

        for attr_name, color, points in zip(attr_names, colors, goal_points):
            line = container.gym_render.create_user_line(
                points,
                color,
                line_width=self.line_width,
                visible=True,
                env_handle=env_handle,
            )
            container.gym_render.register_line_shape(line)
            setattr(self, attr_name, line)

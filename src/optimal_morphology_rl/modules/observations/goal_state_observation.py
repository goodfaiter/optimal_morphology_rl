"""Observation source for the object goal pose in the robot frame."""

from __future__ import annotations

from typing import Any

import torch
from vlearn.torch_utils.torch_jit_utils import quat_conjugate, quat_mul, quat_rotate_inverse

from optimal_morphology_rl.helpers.numpy_vlearn import quaternion_to_6d
from optimal_morphology_rl.modules.observations.observation_base_module import (
    ObservationBaseModule,
)
from optimal_morphology_rl.modules.observations.observation_manager_module import (
    register_observation,
)


@register_observation("goal_state")
class GoalStateObservation(ObservationBaseModule):
    """Goal position and orientation expressed in the robot base frame."""

    def get_obs_dim(self, env: Any) -> int:
        return 3 + 6  # goal pos, 6d goal rot

    def compute_observation(self, env: Any, out: torch.Tensor) -> None:
        container = env.module_manager.container
        robot = container.robot
        reward_object = container.reward_object
        robot_state = robot.get_state()

        quat_robot_to_world = robot_state["quat_robot_to_world"]
        quat_world_to_robot = quat_conjugate(quat_robot_to_world)
        robot_pos_world = robot_state["robot_pos_in_world"]

        goal_pos_world = reward_object.goal_pos_in_world
        quat_object_goal_to_world = reward_object.goal_quat_object_to_world

        object_goal_pos_in_robot_frame = quat_rotate_inverse(
            quat_robot_to_world, goal_pos_world - robot_pos_world
        )
        _6d_object_goal_to_robot = quaternion_to_6d(
            quat_mul(quat_world_to_robot, quat_object_goal_to_world)
        )

        out[:, :3] = object_goal_pos_in_robot_frame
        out[:, 3:9] = _6d_object_goal_to_robot

"""Shared robot-control helpers: active motor mask, action space, and validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from vlearn.spaces import Box

from optimal_morphology_rl.modules.module_container import ModuleContainer


def build_active_dof_mask(container: ModuleContainer, config: dict[str, Any]) -> None:
    """Build the mask and slices used to map policy actions to robot actuators.

    For tendon-driven hands the action vector maps directly to the tendons.
    For motor-driven hands, motors whose names contain any of the substrings in
    ``passive_motor_substrings`` are passive (spring only); everything else is
    active. Alternatively ``active_motor_substrings`` can be provided to
    explicitly select active motors.

    Results are stored directly on ``container``.
    """
    robot = container.robot
    if robot.art_def is None or robot.num_motors is None:
        raise RuntimeError("create_envs must be called before build_active_dof_mask")

    device = robot.motor_to_joint_dof_index.device
    container.root_slice = slice(0, 6) if not robot.fixed_hand else slice(0, 0)

    if robot.use_tendon:
        # Tendon-driven hands: actions map one-to-one to spatial tendons.
        num_active = robot.num_tendons
        container.num_active_dofs = num_active
        container.active_dof_mask = torch.ones(num_active, dtype=torch.bool, device=device)
        container.active_dof_indices = torch.arange(num_active, device=device)
        if robot.fixed_hand:
            container.active_dof_slice = slice(0, num_active)
        else:
            container.active_dof_slice = slice(6, 6 + num_active)
        return

    active_substrings = config.get("active_motor_substrings")
    passive_substrings = config.get("passive_motor_substrings", ["abd"])

    mask = torch.ones(robot.num_motors, dtype=torch.bool, device=device)
    for i in range(robot.num_motors):
        name = robot.art_def.get_motor_def(i).name.lower()
        if active_substrings is not None:
            mask[i] = any(sub.lower() in name for sub in active_substrings)
        else:
            mask[i] = not any(sub.lower() in name for sub in passive_substrings)

    container.active_dof_mask = mask
    container.active_dof_indices = torch.nonzero(mask, as_tuple=False).flatten()
    container.num_active_dofs = int(mask.sum().item())

    if robot.fixed_hand:
        container.active_dof_slice = slice(0, container.num_active_dofs)
    else:
        container.active_dof_slice = slice(6, 6 + container.num_active_dofs)


def get_num_actions(container: ModuleContainer) -> int:
    """Return the number of actions for the robot."""
    robot = container.robot
    num_active = robot.num_tendons if robot.use_tendon else container.num_active_dofs
    return num_active if robot.fixed_hand else 6 + num_active


def build_action_space(container: ModuleContainer) -> None:
    """Build the active motor mask and set the environment action space once."""
    if container.get("num_actions") is not None:
        return
    build_active_dof_mask(container, container.create_robot_config)

    num_actions = get_num_actions(container)
    container.env.action_space = Box(
        low=np.full(num_actions, -1.0, dtype=np.float32),
        high=np.full(num_actions, 1.0, dtype=np.float32),
        dtype=np.float32,
    )
    container.num_actions = num_actions


def validate_robot_dependencies(container: ModuleContainer, name: str) -> None:
    """Ensure the robot and its config are populated by the create_robot module."""
    if container.get("robot") is None:
        raise RuntimeError(
            f"{name} requires 'robot' in the shared container. "
            f"Ensure the 'create_robot' module is listed before '{name}'."
        )
    if container.get("create_robot_config") is None:
        raise RuntimeError(
            f"{name} requires 'create_robot_config' in the shared container. "
            f"Ensure the 'create_robot' module is listed before '{name}'."
        )


def validate_action_buffers(container: ModuleContainer, name: str) -> None:
    """Ensure the action buffers are allocated by process_variable_actions."""
    if container.get("act_buf") is None or container.get("scaled_act_buf") is None:
        raise RuntimeError(
            f"{name} requires 'act_buf' and 'scaled_act_buf' in the shared container. "
            f"Ensure 'process_variable_actions' is listed before '{name}' in pre_physics_step_modules."
        )

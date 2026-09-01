"""Module that dispatches pre-physics control hooks to scene objects."""

from __future__ import annotations

from typing import Any

import torch
import vlearn as v

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.create_objects_module import Button, Drawer
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


def _allocate_control_buffers(obj: Any, total_num_envs: int, device: torch.device) -> None:
    """Allocate control buffers for objects that need them."""
    if isinstance(obj, Drawer):
        if obj.handle_joint_motor_index is not None and obj.drawer_joint_motor_index is not None and obj.num_motors > 0:
            obj.spring_motor_cmd_buf = torch.zeros((total_num_envs, obj.num_motors), device=device, dtype=torch.float32)
            obj.unlocked_buf = torch.zeros((total_num_envs,), device=device, dtype=torch.bool)
    elif isinstance(obj, Button):
        if obj.button_joint_motor_index is not None and obj.num_motors > 0:
            obj.spring_motor_cmd_buf = torch.zeros((total_num_envs, obj.num_motors), device=device, dtype=torch.float32)


def _create_control_gpu_commands(obj: Any, env_group: Any, gym: v.Gym) -> None:
    """Create GPU commands for object control."""
    if isinstance(obj, (Drawer, Button)):
        if obj.spring_motor_cmd_buf is not None and obj.num_motors > 0:
            set_motor_cmd = env_group.create_motor_control_command(
                v.wrap_gpu_buffer(obj.spring_motor_cmd_buf),
                obj.handle,
                (0, obj.num_motors),
            )
            obj.gpu_spring_motor_cmd_array = gym.create_gpu_array([set_motor_cmd])


def _drawer_pre_physics_step(obj: Drawer, gym: v.Gym) -> None:
    """Apply a spring torque to the handle and a lock force to the drawer joint."""
    if (
        obj.handle_joint_dof_index is None
        or obj.handle_joint_motor_index is None
        or obj.drawer_joint_dof_index is None
        or obj.drawer_joint_motor_index is None
        or obj.spring_motor_cmd_buf is None
        or obj.gpu_spring_motor_cmd_array is None
        or obj.unlocked_buf is None
    ):
        return

    q_handle = obj.get_joint_pos_buf[:, obj.handle_joint_dof_index]
    qd_handle = obj.get_joint_vel_buf[:, obj.handle_joint_dof_index]
    torque = -obj.spring_stiffness * (q_handle - obj.spring_rest_angle)
    torque = torch.clamp(torque, -obj.max_spring_torque, obj.max_spring_torque)

    obj.unlocked_buf |= torch.abs(q_handle) >= obj.unlock_angle_threshold

    q_drawer = obj.get_joint_pos_buf[:, obj.drawer_joint_dof_index]
    qd_drawer = obj.get_joint_vel_buf[:, obj.drawer_joint_dof_index]
    lock_force = torch.zeros_like(q_drawer)
    locked = ~obj.unlocked_buf
    if locked.any():
        lock_force[locked] = -obj.lock_stiffness * q_drawer[locked] - obj.lock_damping * qd_drawer[locked]
        lock_force = torch.clamp(lock_force, -obj.max_lock_force, obj.max_lock_force)

    obj.spring_motor_cmd_buf.zero_()
    obj.spring_motor_cmd_buf[:, obj.handle_joint_motor_index] = torque
    obj.spring_motor_cmd_buf[:, obj.drawer_joint_motor_index] = lock_force
    gym.set_motor_forces(obj.gpu_spring_motor_cmd_array)


def _button_pre_physics_step(obj: Button, gym: v.Gym) -> None:
    """Apply a spring force to the button joint."""
    if (
        obj.button_joint_dof_index is None
        or obj.button_joint_motor_index is None
        or obj.spring_motor_cmd_buf is None
        or obj.gpu_spring_motor_cmd_array is None
    ):
        return

    q_button = obj.get_joint_pos_buf[:, obj.button_joint_dof_index]
    qd_button = obj.get_joint_vel_buf[:, obj.button_joint_dof_index]
    force = -obj.spring_stiffness * (q_button - obj.spring_rest_position)
    force = force - obj.spring_damping * qd_button
    force = torch.clamp(force, -obj.max_spring_force, obj.max_spring_force)

    obj.spring_motor_cmd_buf.zero_()
    obj.spring_motor_cmd_buf[:, obj.button_joint_motor_index] = force
    gym.set_motor_forces(obj.gpu_spring_motor_cmd_array)


def _pre_physics_step(obj: Any, gym: v.Gym) -> None:
    """Dispatch pre-physics control to objects that implement it."""
    if isinstance(obj, Drawer):
        _drawer_pre_physics_step(obj, gym)
    elif isinstance(obj, Button):
        _button_pre_physics_step(obj, gym)


@register_module("object_control")
class ObjectControlModule(BaseModule):
    """Calls pre-physics step hooks on objects created by ``create_objects``."""

    def post_finalize(self, container: ModuleContainer) -> None:
        """Allocate control buffers and create control GPU commands."""
        objects = container.get("objects")
        if objects is None:
            return
        total_num_envs = container.total_num_envs
        device = container.device
        for obj in objects.values():
            _allocate_control_buffers(obj, total_num_envs, device)
            _create_control_gpu_commands(obj, container.env_group, container.gym)

    def step(self, container: ModuleContainer) -> None:
        """Dispatch pre-physics hooks to every loaded object."""
        objects = container.get("objects")
        if objects is None:
            return
        for obj in objects.values():
            _pre_physics_step(obj, container.gym)

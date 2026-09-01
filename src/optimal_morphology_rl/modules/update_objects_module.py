"""Module that refreshes scene object state buffers after the physics step."""

from __future__ import annotations

from functools import partial
from typing import Any

import torch
import vlearn as v

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.create_objects_module import (
    LoadedArticulatedObject,
    LoadedRigidObject,
    ObjectBase,
)
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


def _allocate_get_buffers(obj: ObjectBase, total_num_envs: int, device: torch.device) -> None:
    """Allocate buffers used to read object state from simulation."""
    obj.get_trans_object_to_world_buf = torch.zeros(
        (total_num_envs, 7), device=device, dtype=torch.float32
    )
    obj.get_vel_in_world_buf = torch.zeros(
        (total_num_envs, 6), device=device, dtype=torch.float32
    )

    if isinstance(obj, LoadedArticulatedObject):
        obj.get_joint_pos_buf = torch.zeros(
            (total_num_envs, obj.num_joints), device=device, dtype=torch.float32
        )
        obj.get_joint_vel_buf = torch.zeros(
            (total_num_envs, obj.num_joints), device=device, dtype=torch.float32
        )


def _create_get_gpu_commands(obj: ObjectBase, env_group: Any, gym: v.Gym) -> None:
    """Create GPU commands used to read object state."""
    if isinstance(obj, LoadedRigidObject):
        get_kin_cmd = env_group.create_rigid_body_kinematic_state_command(
            v.wrap_gpu_buffer(obj.get_trans_object_to_world_buf),
            v.wrap_gpu_buffer(obj.get_vel_in_world_buf),
            obj.handle,
        )
        obj.gpu_get_object_kin_cmd_array = gym.create_gpu_array([get_kin_cmd])
    elif isinstance(obj, LoadedArticulatedObject):
        get_kin_cmd = env_group.create_articulation_kinematic_state_command(
            v.wrap_gpu_buffer(obj.get_joint_pos_buf),
            v.wrap_gpu_buffer(obj.get_joint_vel_buf),
            v.wrap_gpu_buffer(obj.get_trans_object_to_world_buf),
            v.wrap_gpu_buffer(obj.get_vel_in_world_buf),
            obj.handle,
            (0, obj.num_joints),
            (0, 1),
        )
        obj.gpu_get_object_kin_cmd_array = gym.create_gpu_array([get_kin_cmd])


def _refresh_buffers(obj: ObjectBase, gym: v.Gym) -> None:
    """Refresh object state buffers from simulation."""
    if isinstance(obj, LoadedRigidObject):
        gym.get_rigid_body_kinematic_states(obj.gpu_get_object_kin_cmd_array)
    elif isinstance(obj, LoadedArticulatedObject):
        gym.get_articulation_kinematic_states(obj.gpu_get_object_kin_cmd_array)


@register_module("update_objects")
class UpdateObjectsModule(BaseModule):
    """Refreshes object state buffers for objects created by ``create_objects``."""

    def post_finalize(self, container: ModuleContainer) -> None:
        """Allocate get buffers, create get GPU commands, and bind refresh_buffers."""
        objects = container.get("objects")
        if objects is None:
            return
        total_num_envs = container.total_num_envs
        device = container.device
        for obj in objects.values():
            _allocate_get_buffers(obj, total_num_envs, device)
            _create_get_gpu_commands(obj, container.env_group, container.gym)
            obj.refresh_buffers = partial(_refresh_buffers, obj)

    def step(self, container: ModuleContainer) -> None:
        """Refresh state buffers for every loaded object."""
        objects = container.get("objects")
        if objects is None:
            return
        for obj in objects.values():
            obj.refresh_buffers(container.gym)

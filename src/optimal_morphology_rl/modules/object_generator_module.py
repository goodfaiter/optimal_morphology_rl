"""Object generator module that owns loading, buffers, and lifecycle of scene objects."""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_manager import register_module
from optimal_morphology_rl.modules.objects import OBJECT_REGISTRY, ObjectBase


@register_module("object_generator")
class ObjectGeneratorModule(BaseModule):
    """Loads scene objects, manages their GPU state, and dispatches per-step hooks.

    Expects the shared container to already contain ``env_def`` (populated by
    ``create_rigid_vsim_envs``).  After ``create_rigid_vsim_envs.post_finalize``
    has created the ``env_group``, this module allocates buffers and builds GPU
    commands in its own ``post_finalize``.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.object_names: List[str] = []
        self.objects: Dict[str, ObjectBase] = {}
        self.reward_object_name: str = self.config.get("reward_object", "drawer")
        self.scene_objects: List[str] = list(self.config.get("scene_objects", []))
        self.record_output_path = self.config.get("record_output_path", None)

    def finalize(self, env: Any) -> None:
        """Instantiate and load requested objects into the environment definition."""
        container = env.module_manager.container
        if container.get("env_def") is None:
            raise RuntimeError(
                "ObjectGeneratorModule requires 'env_def' in the shared container. "
                "Ensure create_rigid_vsim_envs is listed before object_generator."
            )

        table_name = "table" if self.record_output_path is None else "table_with_camera"

        object_names: List[str] = [self.reward_object_name]
        for obj in self.scene_objects:
            if obj not in object_names:
                object_names.append(obj)
        if table_name not in object_names:
            object_names.append(table_name)

        self.object_names = object_names
        self.objects = {}
        for obj_name in object_names:
            if obj_name not in OBJECT_REGISTRY:
                raise ValueError(
                    f"Unknown object: {obj_name}. "
                    f"Available: {list(OBJECT_REGISTRY.keys())}"
                )
            obj = OBJECT_REGISTRY[obj_name]()
            obj.load(container.env_def)
            self.objects[obj_name] = obj

        container.objects = self.objects
        container.object_generator = self
        container.reward_object_name = self.reward_object_name
        container.reward_object = self.objects.get(self.reward_object_name)
        container.object_names = self.object_names

    def post_finalize(self, env: Any) -> None:
        """Allocate buffers and create GPU commands now that env_group exists."""
        container = env.module_manager.container
        if container.get("env_group") is None:
            raise RuntimeError(
                "ObjectGeneratorModule requires 'env_group' in the shared container. "
                "Ensure create_rigid_vsim_envs.post_finalize runs before object_generator.post_finalize."
            )
        if container.get("total_num_envs") is None or container.get("device") is None:
            raise RuntimeError(
                "ObjectGeneratorModule requires 'total_num_envs' and 'device' in the shared container."
            )

        total_num_envs = container.total_num_envs
        device = container.device
        reset_buf = env.reset_buf

        for obj in self.objects.values():
            obj.allocate_buffers(total_num_envs, device)
            obj.create_gpu_command(container.env_group, container.gym, reset_buf)

        # Share link offsets for contact modules downstream.
        container.object_link_offsets = {
            name: self._get_object_link_offset(name) for name in self.object_names
        }
        container.reward_object_link_offset = self._get_object_link_offset(
            self.reward_object_name
        )

    def pre_physics_step(self, env: Any) -> None:
        """Dispatch pre-physics hooks to every loaded object."""
        if not self.objects:
            return
        gym = env.module_manager.container.gym
        for obj in self.objects.values():
            obj.pre_physics_step(gym)

    def post_physics_step(self, env: Any) -> None:
        """Dispatch post-physics hooks to every loaded object."""
        if not self.objects:
            return
        gym = env.module_manager.container.gym
        for obj in self.objects.values():
            obj.post_physics_step(gym)

    def reset(self, env: Any) -> None:
        """Reset objects selected by the environment's reset buffer."""
        if not self.objects:
            return
        gym = env.module_manager.container.gym
        reset_buf = env.reset_buf
        if reset_buf.sum() == 0:
            return
        for obj in self.objects.values():
            obj.reset_idx(gym, reset_buf)

    def refresh_buffers(self, env: Any) -> None:
        """Refresh object state buffers from simulation.

        This is not a manager lifecycle hook; environments or other modules can
        call it explicitly when they need fresh object state.
        """
        if not self.objects:
            return
        gym = env.module_manager.container.gym
        for obj in self.objects.values():
            obj.refresh_buffers(gym)

    def get_object(self, name: str) -> ObjectBase:
        """Get a specific object by name."""
        if name not in self.objects:
            raise ValueError(f"Unknown object: {name}.")
        return self.objects[name]

    def get_object_link_offset(self, name: str) -> int:
        """Return link-based offset for the object based on object order."""
        offset = 0
        for obj_name in self.object_names:
            offset += self.objects[obj_name].get_link_offset()
            if obj_name == name:
                return offset
        raise ValueError(f"Unknown object: {name}.")

    def _get_object_link_offset(self, name: str) -> int:
        """Deprecated alias for :meth:`get_object_link_offset`."""
        return self.get_object_link_offset(name)

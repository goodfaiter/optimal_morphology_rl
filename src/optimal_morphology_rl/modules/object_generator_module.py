"""Object generator module that owns loading, buffers, and lifecycle of scene objects."""

from __future__ import annotations

from typing import Any, List

import torch

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_manager import register_module
from optimal_morphology_rl.modules.object_generator import ObjectGenerator


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
        self.reward_object_name: str = self.config.get("reward_object", "drawer")
        self.scene_objects: List[str] = list(self.config.get("scene_objects", []))
        self.record_output_path = self.config.get("record_output_path", None)
        self.generator: ObjectGenerator | None = None

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

        self.generator = ObjectGenerator(object_names)
        self.generator.load(container.env_def)

        container.objects = self.generator.objects
        container.object_generator = self
        container.reward_object_name = self.reward_object_name
        container.reward_object = self.generator.get_object(self.reward_object_name)
        container.object_names = object_names

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

        self.generator.allocate_buffers(total_num_envs, device)
        self.generator.create_gpu_commands(container.env_group, container.gym, reset_buf)

        # Share link offsets for contact modules downstream.
        container.object_link_offsets = {
            name: self.generator.get_object_link_offset(name)
            for name in self.generator.object_names
        }
        container.reward_object_link_offset = self.generator.get_object_link_offset(
            self.reward_object_name
        )

    def pre_physics_step(self, env: Any) -> None:
        """Dispatch pre-physics hooks to every loaded object."""
        if self.generator is None:
            return
        self.generator.pre_physics_step(env.module_manager.container.gym)

    def post_physics_step(self, env: Any) -> None:
        """Dispatch post-physics hooks to every loaded object."""
        if self.generator is None:
            return
        self.generator.post_physics_step(env.module_manager.container.gym)

    def reset(self, env: Any) -> None:
        """Reset objects selected by the environment's reset buffer."""
        if self.generator is None:
            return
        reset_buf = env.reset_buf
        if reset_buf.sum() == 0:
            return
        self.generator.reset_idx(env.module_manager.container.gym, reset_buf)

    def refresh_buffers(self, env: Any) -> None:
        """Refresh object state buffers from simulation.

        This is not a manager lifecycle hook; environments or other modules can
        call it explicitly when they need fresh object state.
        """
        if self.generator is None:
            return
        self.generator.refresh_buffers(env.module_manager.container.gym)

    def get_object(self, name: str):
        """Get a specific object by name."""
        if self.generator is None:
            raise RuntimeError("ObjectGeneratorModule has not been finalized.")
        return self.generator.get_object(name)

    def get_object_link_offset(self, name: str) -> int:
        """Return link-based offset for the object based on object order."""
        if self.generator is None:
            raise RuntimeError("ObjectGeneratorModule has not been finalized.")
        return self.generator.get_object_link_offset(name)

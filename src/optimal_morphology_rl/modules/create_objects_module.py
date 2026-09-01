"""Module that creates scene objects and exposes them on the shared container."""

from __future__ import annotations

from typing import Any, List

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module
from optimal_morphology_rl.modules.object_generator import ObjectGenerator


@register_module("create_objects")
class CreateObjectsModule(BaseModule):
    """Loads scene objects, manages their GPU state, and exposes them on the container.

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

    def finalize(self, container: ModuleContainer) -> None:
        """Instantiate and load requested objects into the environment definition."""
        if container.get("env_def") is None:
            raise RuntimeError(
                "CreateObjectsModule requires 'env_def' in the shared container. "
                "Ensure create_rigid_vsim_envs is listed before create_objects."
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
        container.create_objects = self
        container.reward_object_name = self.reward_object_name
        container.reward_object = self.generator.get_object(self.reward_object_name)
        container.object_names = object_names

    def post_finalize(self, container: ModuleContainer) -> None:
        """Allocate buffers and create GPU commands now that env_group exists."""
        if container.get("env_group") is None:
            raise RuntimeError(
                "CreateObjectsModule requires 'env_group' in the shared container. "
                "Ensure create_rigid_vsim_envs.post_finalize runs before create_objects.post_finalize."
            )
        if container.get("total_num_envs") is None or container.get("device") is None:
            raise RuntimeError(
                "CreateObjectsModule requires 'total_num_envs' and 'device' in the shared container."
            )
        if container.get("reset_buf") is None:
            raise RuntimeError(
                "CreateObjectsModule requires 'reset_buf' in the shared container. "
                "Ensure ModularEnvironment sets container.reset_buf before post_finalize."
            )

        total_num_envs = container.total_num_envs
        device = container.device
        reset_buf = container.reset_buf

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

    def reset(self, container: ModuleContainer) -> None:
        """Reset objects selected by the environment's reset buffer."""
        if self.generator is None:
            return
        reset_buf = container.get("reset_buf")
        if reset_buf is None or reset_buf.sum() == 0:
            return
        self.generator.reset_idx(container.gym, reset_buf)

    def get_object(self, name: str):
        """Get a specific object by name."""
        if self.generator is None:
            raise RuntimeError("CreateObjectsModule has not been finalized.")
        return self.generator.get_object(name)

    def get_object_link_offset(self, name: str) -> int:
        """Return link-based offset for the object based on object order."""
        if self.generator is None:
            raise RuntimeError("CreateObjectsModule has not been finalized.")
        return self.generator.get_object_link_offset(name)

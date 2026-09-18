"""Module that discovers object cameras and records frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import vlearn as v
from PIL import Image

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.create_objects_module import LoadedRigidObject
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@dataclass(frozen=True)
class CameraSpec:
    """Description of a camera found on a loaded object."""

    object_name: str
    kind: str
    definition_name: str
    instance_name: str
    resolution_x: int
    resolution_y: int
    far_clip: float


@dataclass
class CameraBinding:
    """Runtime state for a bound camera."""

    spec: CameraSpec
    buffer: torch.Tensor
    command: object
    command_array: object


class ObjectCameraRecorder:
    """Discover cameras on loaded objects and save them to disk."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._specs: list[CameraSpec] = []
        self._bindings: list[CameraBinding] = []

    def update(self, gym: Any) -> None:
        """Refresh camera buffers from simulation."""
        rgb_bindings = [binding for binding in self._bindings if binding.spec.kind == "rgb"]
        for binding in rgb_bindings:
            gym.get_rgb_camera_images(binding.command_array)

    def save(self, frame_index: int) -> None:
        """Save PNGs for all bound cameras."""
        for binding in self._bindings:
            self._save_binding(binding, frame_index)

    def build_specs(self, loaded_objects: dict, env_def) -> None:
        """Discover camera specs from the loaded objects mapping (container.objects)."""
        for obj in loaded_objects.values():
            if isinstance(obj, LoadedRigidObject):
                self._build_spec(obj, env_def=env_def)

    def build_cameras(
        self,
        env_def,
        env_group,
        gym: Any,
        device: torch.device,
    ) -> None:
        """Instantiate camera buffers and GPU commands."""
        for spec in self._specs:
            self._bindings.append(self._build_camera(spec, env_def, env_group, gym, device))

    def _build_spec(self, loaded_object, env_def) -> None:
        rigid_body_def = env_def.get_rigid_body_def_by_name(loaded_object.name)

        num_camera_defs = rigid_body_def.get_num_rgb_camera_defs()
        num_camera_instances = rigid_body_def.get_num_rgb_cameras()
        for def_index in range(num_camera_defs):
            for instance_index in range(num_camera_instances):
                self._specs.append(self._build_rgb_spec(rigid_body_def, loaded_object.name, def_index, instance_index))

    def _build_camera(
        self,
        spec: CameraSpec,
        env_def,
        env_group,
        gym: Any,
        device: torch.device,
    ) -> CameraBinding:
        rigid_body_handle = env_def.get_rigid_body_handle_by_name(spec.object_name)
        rigid_body = env_def.get_rigid_body(rigid_body_handle)
        return self._bind_rgb_camera(spec, rigid_body, env_group, gym, device)

    def _build_rgb_spec(
        self,
        rigid_body_def,
        object_name: str,
        def_index: int,
        instance_index: int,
    ) -> CameraSpec:
        def_name = rigid_body_def.get_rgb_camera_def_name(def_index)
        instance_name = rigid_body_def.get_rgb_camera_name(instance_index)
        camera_def = rigid_body_def.get_rgb_camera_def_by_name(def_name)
        camera = rigid_body_def.get_rgb_camera_by_name(instance_name)
        camera.render_relative_transform = camera.relative_transform
        camera.render_width = 1
        camera.render_height = 1
        camera.base_color = v.Vec3(1.0, 1.0, 1.0)
        return CameraSpec(
            object_name=object_name,
            kind="rgb",
            definition_name=def_name,
            instance_name=instance_name,
            resolution_x=camera_def.resolution_x,
            resolution_y=camera_def.resolution_y,
            far_clip=camera_def.far_clip,
        )

    def _bind_rgb_camera(
        self,
        spec: CameraSpec,
        rigid_body,
        env_group,
        gym: Any,
        device: torch.device,
    ) -> CameraBinding:
        total_envs = env_group.get_total_num_environments()
        camera_handle = rigid_body.get_rgb_camera_handle_by_name(spec.instance_name)
        # The command buffers are sized by the camera's resolution as seen by the
        # environment group, which may differ from the camera def resolution.
        resolution_x, resolution_y = env_group.get_rgb_camera_resolution(camera_handle)
        buffer = torch.empty(
            (total_envs, resolution_y, resolution_x, 4),
            dtype=torch.uint8,
            device=device,
        )
        command = env_group.create_rgb_camera_command(v.wrap_gpu_buffer(buffer), camera_handle)
        command_array = gym.create_gpu_array([command])
        return CameraBinding(spec=spec, buffer=buffer, command=command, command_array=command_array)

    def _save_binding(self, binding: CameraBinding, frame_index: int) -> Path:
        spec = binding.spec
        object_dir = self.output_dir / spec.object_name / spec.instance_name
        object_dir.mkdir(parents=True, exist_ok=True)

        filename = f"step_{frame_index:06d}.png"
        output_path = object_dir / filename
        print(f"Saving camera image to {output_path}")
        image = binding.buffer[0].detach().cpu().numpy()
        if image.shape[-1] == 4:
            pil_image = Image.fromarray(np.ascontiguousarray(image), mode="RGBA")
        else:
            pil_image = Image.fromarray(np.ascontiguousarray(image[:, :, :3]), mode="RGB")
        pil_image.save(output_path)
        return output_path


@register_module("camera_recorder")
class CameraRecorderModule(BaseModule):
    """Wraps ObjectCameraRecorder and records frames for the configured number of steps."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        output_dir = self.config.get("output_dir")
        self.recorder: ObjectCameraRecorder | None = None
        self._num_steps: int | None = self.config.get("num_steps")
        self._frames_recorded = 0
        if output_dir is not None:
            self.recorder = ObjectCameraRecorder(output_dir)

    def finalize(self, container: ModuleContainer) -> None:
        """Build camera specs from the loaded objects."""
        if self.recorder is None:
            return
        if container.get("objects") is None or container.get("env_def") is None:
            raise RuntimeError("CameraRecorderModule requires 'objects' and 'env_def' in the shared container.")
        self.recorder.build_specs(container.objects, container.env_def)

    def post_finalize(self, container: ModuleContainer) -> None:
        """Create camera instances on the environment group."""
        if self.recorder is None:
            return
        self.recorder.build_cameras(
            container.env_def,
            container.env_group,
            container.gym,
            container.device,
        )
        container.camera_recorder = self.recorder

    def step(self, container: ModuleContainer) -> None:
        """Record a frame from each camera until the configured number of steps is reached."""
        if self.recorder is None:
            return
        if self._num_steps is not None and self._frames_recorded >= self._num_steps:
            return
        self.recorder.update(container.gym)
        self.recorder.save(self._frames_recorded)
        self._frames_recorded += 1

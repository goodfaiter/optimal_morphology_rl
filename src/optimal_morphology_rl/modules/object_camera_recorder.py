from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import vlearn as v


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
    
    def update(self, gym) -> None:
        """Refresh camera buffers from simulation."""
        rgb_bindings = [binding for binding in self._bindings if binding.spec.kind == "rgb"]
        for binding in rgb_bindings:
            gym.get_rgb_camera_images(binding.command_array)

    def save(self, timestep: int) -> None:
        """Refresh camera buffers and save PNGs for the requested env indices."""
        for binding in self._bindings:
            self._save_binding(binding, timestep)

    def build_specs(self, object_generator, env_def) -> None:
        for obj in object_generator.objects.values():
            self._build_spec(obj, env_def=env_def)

    def build_cameras(self, env_def, env_group, gym, num_envs: int, device: torch.device) -> None:
        for spec in self._specs:
            self._bindings.append(self._build_camera(spec, env_def, env_group, gym, num_envs, device))

    def _build_spec(self, loaded_object, env_def) -> None:

        rigid_body_def = env_def.get_rigid_body_def_by_name(loaded_object.name)

        for index in range(rigid_body_def.get_num_rgb_camera_defs()):
            self._specs.append(self._build_rgb_spec(rigid_body_def, loaded_object.name, index))

    def _build_camera(self, spec: CameraSpec, env_def, env_group, gym, num_envs: int, device: torch.device) -> CameraBinding:
            rigid_body_handle = env_def.get_rigid_body_handle_by_name(spec.object_name)
            rigid_body = env_def.get_rigid_body(rigid_body_handle)
            return self._bind_rgb_camera(spec, rigid_body, env_group, gym, num_envs, device)

    def _build_rgb_spec(self, rigid_body_def, object_name: str, index: int) -> CameraSpec:
        def_name = rigid_body_def.get_rgb_camera_def_name(index)
        instance_name = rigid_body_def.get_rgb_camera_name(index)
        camera_def = rigid_body_def.get_rgb_camera_def_by_name(def_name)
        camera = rigid_body_def.get_rgb_camera_by_name(instance_name)
        camera.render_relative_transform = camera.relative_transform
        camera.render_width = 1
        camera.render_height = 1
        return CameraSpec(
            object_name=object_name,
            kind="rgb",
            definition_name=def_name,
            instance_name=instance_name,
            resolution_x=camera_def.resolution_x,
            resolution_y=camera_def.resolution_y,
            far_clip=camera_def.far_clip,
        )

    def _bind_rgb_camera(self, spec: CameraSpec, rigid_body, env_group, gym, num_envs: int, device: torch.device) -> CameraBinding:
        buffer = torch.empty(
            (len(num_envs), spec.resolution_y, spec.resolution_x, 4),
            dtype=torch.uint8,
            device=device,
        )
        camera_handle = rigid_body.get_rgb_camera_handle_by_name(spec.instance_name)
        command = env_group.create_rgb_camera_command(v.wrap_gpu_buffer(buffer), camera_handle)
        command_array = gym.create_gpu_array([command])
        return CameraBinding(spec=spec, buffer=buffer, command=command, command_array=command_array)

    def _save_binding(self, binding: CameraBinding, timestep: int) -> Path:
        spec = binding.spec
        object_dir = self.output_dir / spec.object_name / spec.instance_name
        object_dir.mkdir(parents=True, exist_ok=True)

        filename = f"step_{timestep:06d}.png"
        output_path = object_dir / filename
        print(f"Saving camera image to {output_path}")
        image = binding.buffer[0].detach().cpu().numpy()
        if image.shape[-1] == 4:
            pil_image = Image.fromarray(np.ascontiguousarray(image), mode="RGBA")
        else:
            pil_image = Image.fromarray(np.ascontiguousarray(image[:, :, :3]), mode="RGB")
        pil_image.save(output_path)

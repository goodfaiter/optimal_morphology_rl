"""Module that discovers object cameras and records frames (and videos)."""

from __future__ import annotations

import atexit
import dataclasses
import importlib.resources as resources
import subprocess
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
    # vlearn's RGB images are bottom-up (OpenGL rows): flip them when saving. The camera rig's cameras are posed
    # upright and flipped; object cameras (e.g. table_with_camera) keep their own, compensated poses.
    flip_vertical: bool = False


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
                self._build_spec(obj.name, env_def=env_def)

    def build_body_specs(self, body_name: str, env_def, flip_vertical: bool = False) -> None:
        """Discover camera specs of a rigid body created by name (e.g. the camera rig)."""
        self._build_spec(body_name, env_def=env_def, flip_vertical=flip_vertical)

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

    def _build_spec(self, object_name: str, env_def, flip_vertical: bool = False) -> None:
        rigid_body_def = env_def.get_rigid_body_def_by_name(object_name)

        num_camera_defs = rigid_body_def.get_num_rgb_camera_defs()
        num_camera_instances = rigid_body_def.get_num_rgb_cameras()
        for def_index in range(num_camera_defs):
            for instance_index in range(num_camera_instances):
                spec = self._build_rgb_spec(rigid_body_def, object_name, def_index, instance_index)
                self._specs.append(dataclasses.replace(spec, flip_vertical=flip_vertical))

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
        image = binding.buffer[0].detach().cpu().numpy()
        if spec.flip_vertical:
            image = image[::-1]
        if image.shape[-1] == 4:
            pil_image = Image.fromarray(np.ascontiguousarray(image), mode="RGBA")
        else:
            pil_image = Image.fromarray(np.ascontiguousarray(image[:, :, :3]), mode="RGB")
        pil_image.save(output_path)
        return output_path


def _camera_rig_path(name: str) -> str:
    return str(resources.files("optimal_morphology_rl_assets.assets") / f"objects/{name}.vsim")


@register_module("camera_recorder")
class CameraRecorderModule(BaseModule):
    """Records PNG frames of the RGB cameras and turns each camera's frames into an MP4 (ffmpeg).

    Cameras come from the loaded objects (e.g. ``table_with_camera``) and, with ``camera_rig``, from a fixed,
    collision-free body this module loads at the env origin (``assets/objects/<camera_rig>.vsim``), so a task needs
    no extra object. Frames of env 0 go to ``<output_dir>/<body>/<camera>/step_%06d.png`` every ``frame_stride``
    control steps; after ``num_steps`` frames (or at exit) each camera's frames become ``<video_prefix><camera>.mp4``
    in ``video_dir`` (default: next to the frames), at the real-time frame rate. In play mode the runner sets, when
    not given, ``output_dir`` to ``<run>/videos/<checkpoint>_<time>`` and the videos to the run folder:
    ``<run>/<checkpoint>_<time>_<camera>.mp4``.

    Config::

        camera_recorder:
          enabled: ${eval:'mode == "play"'}   # e.g. only when playing a policy
          camera_rig: camera_rig              # optional
          frame_stride: 7                     # optional, default 1 (every control step); 7 = ~30 fps at 200 Hz
          num_steps: 600                      # optional: frames to record, then write the videos
          output_dir: ...                     # optional in play mode (frames)
          video_dir: ...                      # optional, where the .mp4 files go
          video_prefix: ...                   # optional, prepended to <camera>.mp4
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.enabled = bool(self.config.get("enabled", True))
        output_dir = self.config.get("output_dir")
        self.recorder: ObjectCameraRecorder | None = None
        self._num_steps: int | None = self.config.get("num_steps")
        self._frames_recorded = 0
        self._step_count = 0
        self._videos_written = False
        self.rig_name: str | None = self.config.get("camera_rig")
        self.video_dir: str | None = self.config.get("video_dir")
        self.video_prefix = str(self.config.get("video_prefix", ""))
        self.frame_stride = int(self.config.get("frame_stride", 1))
        self.fps = 30.0  # set from the control step in post_finalize
        if self.enabled and output_dir is not None:
            self.recorder = ObjectCameraRecorder(output_dir)

    def finalize(self, container: ModuleContainer) -> None:
        """Load the camera rig and build camera specs from the loaded objects."""
        if self.recorder is None:
            return
        if container.get("env_def") is None:
            raise RuntimeError("CameraRecorderModule requires 'env_def' in the shared container.")
        if self.rig_name is not None:
            env_def = container.env_def
            env_def.import_definitions(_camera_rig_path(self.rig_name), fixed=True, use_visual_mesh=True)
            handle = env_def.get_rigid_body_def_handle_by_name(self.rig_name)
            env_def.create_rigid_body(handle, v.Transform(v.Quat(0, 0, 0, 1), v.Vec3(0, 0, 0)), self.rig_name)
            self.recorder.build_body_specs(self.rig_name, env_def, flip_vertical=True)
        if container.get("objects") is not None:
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
        dt = float(container.timestep) * float(container.frame_skip)
        self.fps = 1.0 / (dt * self.frame_stride)
        names = [f"{b.spec.object_name}/{b.spec.instance_name}" for b in self.recorder._bindings]
        print(f"CameraRecorder: {names} -> {self.recorder.output_dir} every {self.frame_stride} steps ({self.fps:.1f} fps)")
        atexit.register(self.write_videos)

    def step(self, container: ModuleContainer) -> None:
        """Record a frame from each camera every frame_stride steps until num_steps frames are recorded."""
        if self.recorder is None or self._videos_written:
            return
        self._step_count += 1
        if (self._step_count - 1) % self.frame_stride:
            return
        self.recorder.update(container.gym)
        self.recorder.save(self._frames_recorded)
        self._frames_recorded += 1
        if self._num_steps is not None and self._frames_recorded >= self._num_steps:
            self.write_videos()

    def write_videos(self) -> None:
        """Encode each camera's PNG frames into <camera>.mp4 next to them (as rl_pipeline.create_videos)."""
        if self.recorder is None or self._videos_written or self._frames_recorded == 0:
            return
        self._videos_written = True
        for binding in self.recorder._bindings:
            img_dir = self.recorder.output_dir / binding.spec.object_name / binding.spec.instance_name
            mp4_dir = Path(self.video_dir) if self.video_dir else img_dir
            mp4_dir.mkdir(parents=True, exist_ok=True)
            mp4_path = mp4_dir / f"{self.video_prefix}{binding.spec.instance_name}.mp4"
            cmd = ["ffmpeg", "-y", "-framerate", f"{self.fps:.4f}", "-i", str(img_dir / "step_%06d.png"),
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4_path)]
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                print(f"CameraRecorder: wrote {mp4_path} ({self._frames_recorded} frames)")
            else:
                print(f"CameraRecorder: ffmpeg failed for {mp4_path}: {result.stderr[-500:]}")

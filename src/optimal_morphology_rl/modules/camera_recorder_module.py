"""Module that discovers object cameras and records frames."""

from __future__ import annotations

from typing import Any

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_manager import register_module
from optimal_morphology_rl.modules.object_camera_recorder import ObjectCameraRecorder


@register_module("camera_recorder")
class CameraRecorderModule(BaseModule):
    """Wraps ObjectCameraRecorder and records frames each control step."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        output_dir = self.config.get("output_dir")
        self.recorder: ObjectCameraRecorder | None = None
        if output_dir is not None:
            self.recorder = ObjectCameraRecorder(output_dir)

    def finalize(self, env: Any) -> None:
        if self.recorder is None:
            return
        container = env.module_manager.container
        if container.get("objects") is None or container.get("env_def") is None:
            raise RuntimeError(
                "CameraRecorderModule requires 'objects' and 'env_def' in the shared container."
            )
        self.recorder.build_specs(container.objects, container.env_def)

    def post_finalize(self, env: Any) -> None:
        if self.recorder is None:
            return
        container = env.module_manager.container
        self.recorder.build_cameras(
            container.env_def,
            container.env_group,
            container.gym,
            env.num_envs,
            env.device,
        )
        container.camera_recorder = self.recorder

    def post_physics_step(self, env: Any) -> None:
        if self.recorder is None:
            return
        gym = env.module_manager.container.gym
        self.recorder.update(gym)
        self.recorder.save(env.progress_buf[0].cpu().item())

"""Goal-pose module whose goals move continuously along per-env random sinusoids."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.goal_pose_base_module import GoalPoseBaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("goal_pose_sinusoidal")
class GoalPoseSinusoidalModule(GoalPoseBaseModule):
    """Moves the goal position along a per-env random sinusoid per axis.

    Every reset draws a new pattern: the amplitude is a random fraction of
    each axis's half-range (``min/max_amplitude_fraction``), the frequency is
    random in ``[min_frequency, max_frequency]`` (Hz), and the phase is random
    in ``[0, 2pi)``. The position is updated every step as
    ``center + amp * sin(2pi * freq * t + phase)`` with the control timestep
    ``container.timestep * container.frame_skip``, so the goal moves
    continuously and never leaves the configured ranges. Orientation behaves
    as in the discrete module (resampled on reset).

    Config shape::

        goal_pose_sinusoidal:
          goal_position:
            x: [-0.09, 0.0]
            y: [-0.12, -0.12]
            z: [0.15, 0.20]
          goal_orientation: "random"  # or a fixed [x, y, z, w] quaternion
          sinusoidal:
            min_frequency: 0.2
            max_frequency: 0.8
            min_amplitude_fraction: 0.3
            max_amplitude_fraction: 1.0
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        sinusoidal_config = self.config.get("sinusoidal", {})
        self.min_frequency = float(sinusoidal_config.get("min_frequency", 0.2))
        self.max_frequency = float(sinusoidal_config.get("max_frequency", 0.8))
        self.min_amplitude_fraction = float(sinusoidal_config.get("min_amplitude_fraction", 0.3))
        self.max_amplitude_fraction = float(sinusoidal_config.get("max_amplitude_fraction", 1.0))
        if self.min_frequency <= 0.0 or self.min_frequency > self.max_frequency:
            raise ValueError("goal_pose_sinusoidal requires 0 < min_frequency <= max_frequency (Hz).")
        if not 0.0 <= self.min_amplitude_fraction <= self.max_amplitude_fraction <= 1.0:
            raise ValueError(
                "goal_pose_sinusoidal requires 0 <= min_amplitude_fraction <= max_amplitude_fraction <= 1."
            )
        self._two_pi = 2.0 * torch.pi

    def _allocate(self, container: ModuleContainer) -> None:
        """Allocate the goal buffers and the sinusoid state."""
        super()._allocate(container)
        timestep = container.get("timestep")
        frame_skip = container.get("frame_skip")
        if timestep is None or frame_skip is None:
            raise RuntimeError(
                "goal_pose_sinusoidal requires 'timestep' and 'frame_skip' in the shared "
                "container. Ensure create_rigid_vsim_envs is listed before it."
            )
        lo = (self.x_range[0], self.y_range[0], self.z_range[0])
        hi = (self.x_range[1], self.y_range[1], self.z_range[1])
        self._center = torch.tensor([(l + h) / 2.0 for l, h in zip(lo, hi)], device=container.device, dtype=torch.float32)
        self._half_range = torch.tensor(
            [abs(h - l) / 2.0 for l, h in zip(lo, hi)], device=container.device, dtype=torch.float32
        )
        self._dt = float(timestep) * float(frame_skip)
        self._time_buf = torch.zeros(container.total_num_envs, device=container.device, dtype=torch.float32)
        self._freq = torch.zeros((container.total_num_envs, 3), device=container.device, dtype=torch.float32)
        self._phase = torch.zeros((container.total_num_envs, 3), device=container.device, dtype=torch.float32)
        self._amp = torch.zeros((container.total_num_envs, 3), device=container.device, dtype=torch.float32)

    def _resample(self, container: ModuleContainer, mask: torch.Tensor | None) -> None:
        """Draw a new random sinusoid pattern for the envs selected by ``mask`` (all if None)."""
        self._sample_sinusoid(container, mask)

    def _sample_sinusoid(self, container: ModuleContainer, mask: torch.Tensor | None) -> None:
        """Draw a new random sinusoid pattern and set the goal to its t=0 pose."""
        pos_buf = container.pos_goal_in_world
        num = pos_buf.shape[0] if mask is None else int(mask.sum().item())
        device = pos_buf.device

        fraction = self.min_amplitude_fraction + (self.max_amplitude_fraction - self.min_amplitude_fraction) * torch.rand(
            (num, 3), device=device
        )
        amp = fraction * self._half_range
        freq = self.min_frequency + (self.max_frequency - self.min_frequency) * torch.rand((num, 3), device=device)
        phase = self._two_pi * torch.rand((num, 3), device=device)

        if mask is None:
            self._amp[:] = amp
            self._freq[:] = freq
            self._phase[:] = phase
            self._time_buf[:] = 0.0
        else:
            self._amp[mask] = amp
            self._freq[mask] = freq
            self._phase[mask] = phase
            self._time_buf[mask] = 0.0

        pos = self._center + amp * torch.sin(phase)
        self._sample_quat(container.quat_goal_to_world, mask)

        if mask is None:
            pos_buf[:] = self._to_frame(pos)
        else:
            pos_buf[mask] = self._to_frame(pos, mask)

    def step(self, container: ModuleContainer) -> None:
        """Advance the sinusoid one control step and write the goal positions."""
        if "pos_goal_in_world" not in container:
            return
        self._time_buf += self._dt
        t = self._time_buf.unsqueeze(-1)
        container.pos_goal_in_world[:] = self._to_frame(self._center + self._amp * torch.sin(self._two_pi * self._freq * t + self._phase))

"""Goal-pose module whose goals track continuously along a square's outer edge."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.goal_pose_base_module import GoalPoseBaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("goal_pose_square")
class GoalPoseSquareModule(GoalPoseBaseModule):
    """Moves the goal position along the outer edge of a planar square.

    The square spans the two axes selected by ``plane`` (axis indices 0=x,
    1=y, 2=z); by default the two axes with the largest half-ranges. The
    corners are the extremes of those axes (the inactive axis stays at the
    center of its range). Every reset draws a new pattern: the speed is
    random in ``[min_speed, max_speed]`` (m/s) and the start position is a
    random offset along the perimeter (`[0, 2 * (w + h))`). The position
    advances every step at that speed along the perimeter (piecewise-linear
    between corners), so the goal moves continuously along the outer edge and
    never leaves the configured ranges. Orientation behaves as in the
    discrete module (resampled on reset).

    Config shape::

        goal_pose_square:
          goal_position:
            x: [-0.09, 0.0]
            y: [-0.12, -0.12]
            z: [0.15, 0.20]
          goal_orientation: "random"  # or a fixed [x, y, z, w] quaternion
          plane: [0, 2]               # optional: axis indices of the square's plane
          speed:
            min: 0.01
            max: 0.05
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        speed_config = self.config.get("speed", {})
        self.min_speed = float(speed_config.get("min", 0.01))
        self.max_speed = float(speed_config.get("max", 0.05))
        if self.min_speed <= 0.0 or self.min_speed > self.max_speed:
            raise ValueError("goal_pose_square requires 0 < min_speed <= max_speed (m/s).")

        plane = self.config.get("plane")
        if plane is None:
            half_ranges = [abs(hi - lo) / 2.0 for lo, hi in (self.x_range, self.y_range, self.z_range)]
            order = sorted(range(3), key=lambda i: half_ranges[i], reverse=True)
            self._plane = (order[0], order[1])
        else:
            axes = tuple(int(v) for v in plane)
            if len(axes) != 2 or len(set(axes)) != 2 or not all(0 <= i < 3 for i in axes):
                raise ValueError(
                    "goal_pose_square config 'plane' must be a list of two distinct axis indices in [0, 2]."
                )
            self._plane = axes

    def _allocate(self, container: ModuleContainer) -> None:
        """Allocate the goal buffers and the square-path state."""
        super()._allocate(container)
        timestep = container.get("timestep")
        frame_skip = container.get("frame_skip")
        if timestep is None or frame_skip is None:
            raise RuntimeError(
                "goal_pose_square requires 'timestep' and 'frame_skip' in the shared "
                "container. Ensure create_rigid_vsim_envs is listed before it."
            )
        axis0, axis1 = self._plane
        axis2 = ({0, 1, 2} - {axis0, axis1}).pop()
        ranges = (self.x_range, self.y_range, self.z_range)
        lo0, hi0 = ranges[axis0]
        lo1, hi1 = ranges[axis1]
        lo2, hi2 = ranges[axis2]
        device = container.device

        w = abs(hi0 - lo0)
        h = abs(hi1 - lo1)
        corners = [
            [min(lo0, hi0), min(lo1, hi1)],
            [max(lo0, hi0), min(lo1, hi1)],
            [max(lo0, hi0), max(lo1, hi1)],
            [min(lo0, hi0), max(lo1, hi1)],
            [min(lo0, hi0), min(lo1, hi1)],
        ]
        self._axis0 = axis0
        self._axis1 = axis1
        self._axis2 = axis2
        self._corners = torch.tensor(corners, device=device, dtype=torch.float32)
        self._cum_bounds = torch.tensor([0.0, w, w + h, 2 * w + h, 2 * (w + h)], device=device, dtype=torch.float32)
        self._perimeter = float(2 * (w + h))
        self._center_inactive = torch.tensor((lo2 + hi2) / 2.0, device=device, dtype=torch.float32)
        self._dt = float(timestep) * float(frame_skip)
        self._time_buf = torch.zeros(container.total_num_envs, device=device, dtype=torch.float32)
        self._speed = torch.zeros(container.total_num_envs, device=device, dtype=torch.float32)
        self._start = torch.zeros(container.total_num_envs, device=device, dtype=torch.float32)

    def _resample(self, container: ModuleContainer, mask: torch.Tensor | None) -> None:
        """Draw a new speed/start offset for the envs selected by ``mask`` (all if None)."""
        pos_buf = container.pos_goal_in_world
        num = pos_buf.shape[0] if mask is None else int(mask.sum().item())
        device = pos_buf.device

        speed = self.min_speed + (self.max_speed - self.min_speed) * torch.rand((num,), device=device)
        start = self._perimeter * torch.rand((num,), device=device)

        if mask is None:
            self._speed[:] = speed
            self._start[:] = start
            self._time_buf[:] = 0.0
        else:
            self._speed[mask] = speed
            self._start[mask] = start
            self._time_buf[mask] = 0.0

        distance = self._start if mask is None else self._start[mask]
        self._sample_quat(container.quat_goal_to_world, mask)

        if mask is None:
            pos_buf[:] = self._positions(distance)
        else:
            pos_buf[mask] = self._positions(distance)

    def _positions(self, distance: torch.Tensor) -> torch.Tensor:
        """Map a distance along the perimeter to goal positions ``(N, 3)``."""
        d = distance % self._perimeter
        idx = (d.unsqueeze(-1) >= self._cum_bounds[1:4]).sum(-1)
        qs = self._corners[idx]
        qe = self._corners[idx + 1]
        seg_start = self._cum_bounds[idx]
        seg_end = self._cum_bounds[idx + 1]
        frac = ((d - seg_start) / (seg_end - seg_start + 1e-12)).unsqueeze(-1)
        plane_pos = qs + frac * (qe - qs)

        out = torch.empty((distance.shape[0], 3), device=distance.device, dtype=torch.float32)
        out[:, self._axis0] = plane_pos[:, 0]
        out[:, self._axis1] = plane_pos[:, 1]
        out[:, self._axis2] = self._center_inactive
        return out

    def step(self, container: ModuleContainer) -> None:
        """Advance the goal along the perimeter for one control step."""
        if "pos_goal_in_world" not in container:
            return
        self._time_buf += self._dt
        distance = self._start + self._speed * self._time_buf
        container.pos_goal_in_world[:] = self._positions(distance)

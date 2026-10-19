"""Goal-pose module that splits the envs evenly over discrete, sinusoidal and square goal types."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from typing import Any

import torch
import vlearn as v
from vlearn.torch_utils.torch_jit_utils import quat_mul, quat_rotate

from optimal_morphology_rl.helpers.numpy_vlearn import random_uniform_quaternion
from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


def _rpy_to_quat(rpy: list[float]) -> list[float]:
    """URDF ``rpy`` (fixed-axis x, then y, then z) to an ``[x, y, z, w]`` quaternion."""
    cr, sr = math.cos(rpy[0] / 2), math.sin(rpy[0] / 2)
    cp, sp = math.cos(rpy[1] / 2), math.sin(rpy[1] / 2)
    cy, sy = math.cos(rpy[2] / 2), math.sin(rpy[2] / 2)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


def goal_in_env_frame(container: ModuleContainer) -> tuple[torch.Tensor, torch.Tensor]:
    """Goal position ``(N, 3)`` and orientation ``(N, 4)`` in the environment frame.

    'goal_pose' keeps the goal in its goal frame (``container.goal_position_in_goal_frame`` /
    ``goal_orientation_in_goal_frame``) together with the frame (``container.goal_frame_quat`` /
    ``goal_frame_pos``); modules that compare it with simulator poses (rewards, terminations,
    visualization) convert it with this.
    """
    pos = container.get("goal_position_in_goal_frame")
    if pos is None:
        raise RuntimeError("The goal is missing: add the goal_pose module (it fills container.goal_position_in_goal_frame).")
    frame_quat = container.goal_frame_quat
    world_pos = quat_rotate(frame_quat, pos) + container.goal_frame_pos
    world_quat = quat_mul(frame_quat, container.goal_orientation_in_goal_frame)
    return world_pos, world_quat


def _frame_chain(vsim_path: str, frame: str, sim_links: list[str]) -> tuple[str, list[dict[str, list[float]]]]:
    """Resolve a robot-description link to its nearest simulated ancestor and the joint origins to it.

    vlearn merges links attached by fixed joints into their parent, so e.g. ``Pinky_Abd`` is not a
    simulated link. Walks the vsim joints from ``frame`` up to the first link vlearn simulates and
    returns that link and the joint origins (``xyz``/``rpy``, parent to child) from it to ``frame``.
    """
    root = ET.parse(vsim_path).getroot()
    links = {link.get("name") for link in root.iter("link")}
    if frame not in links:
        raise RuntimeError(f"goal_pose 'frame' {frame!r} is not a link of {vsim_path}.")
    parent_joint = {joint.find("child").get("link"): joint for joint in root.iter("joint")}
    chain: list[dict[str, list[float]]] = []
    link = frame
    while link not in sim_links:
        joint = parent_joint.get(link)
        if joint is None:
            raise RuntimeError(f"goal_pose 'frame' {frame!r} has no simulated ancestor among {sim_links}.")
        origin = joint.find("origin")
        xyz = [float(x) for x in origin.get("xyz", "0 0 0").split()] if origin is not None else [0.0, 0.0, 0.0]
        rpy = [float(x) for x in origin.get("rpy", "0 0 0").split()] if origin is not None else [0.0, 0.0, 0.0]
        chain.insert(0, {"xyz": xyz, "rpy": rpy})
        link = joint.find("parent").get("link")
    return link, chain


class _GoalType:
    """One goal type. Holds state for all envs but only writes the envs in the masks it is given.

    Type config (shared ``goal_pose`` keys merged with the type's block under ``types``)::

        goal_position: {x: [lo, hi], y: [lo, hi], z: [lo, hi]}
        goal_orientation: "random"  # or a fixed [x, y, z, w] quaternion
        frame: Pinky_Abd            # optional, see below

    Goals are sampled in the goal frame and written as is to ``container.goal_position_in_goal_frame``
    / ``goal_orientation_in_goal_frame``. With ``frame`` the goal frame is that robot-description
    link's frame at the zero pose (read once on the first reset, right after 'create_robot' reset the
    articulation); the link may be one that vlearn merges away (fixed joints): its pose is then
    composed from its nearest simulated ancestor and the vsim joint origins (see ``_frame_chain``).
    Without ``frame`` the goal frame is the environment frame.
    """

    name = ""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        position_config = config.get("goal_position", {})
        ranges = [position_config.get(axis, [0.0, 0.0]) for axis in ("x", "y", "z")]
        self.x_range, self.y_range, self.z_range = ((float(r[0]), float(r[1])) for r in ranges)
        self.goal_orientation = config.get("goal_orientation", "random")
        self.frame = config.get("frame")

    def allocate(self, container: ModuleContainer) -> None:
        """Allocate per-env state."""

    def read_frame(self, container: ModuleContainer) -> tuple[torch.Tensor, torch.Tensor]:
        """Pose of ``frame`` in every env's environment frame, ``(quat (N, 4), pos (N, 3))``, from the simulator and the vsim."""
        robot = container.get("robot")
        vsim_path = container.get("robot_vsim_path")
        if robot is None or vsim_path is None:
            raise RuntimeError("goal_pose 'frame' requires 'robot' and 'robot_vsim_path' in the shared container.")
        links = [robot.art_def.get_link_def(i).name for i in range(robot.num_links)]
        sim_link, chain = _frame_chain(vsim_path, self.frame, links)
        index = links.index(sim_link)
        pose = torch.zeros((container.total_num_envs, 7), device=container.device, dtype=torch.float32)
        command = container.env_group.create_link_transform_command(v.wrap_gpu_buffer(pose), robot.arti_handle, (index, index + 1))
        container.gym.compute_kinematics()
        container.gym.get_link_transforms(container.gym.create_gpu_array([command]))
        quat, pos = pose[:, :4].clone(), pose[:, 4:7].clone()
        for origin in chain:
            offset_pos = torch.tensor(origin["xyz"], device=pos.device, dtype=torch.float32).expand_as(pos)
            offset_quat = torch.tensor(_rpy_to_quat(origin["rpy"]), device=pos.device, dtype=torch.float32)
            pos = pos + quat_rotate(quat, offset_pos)
            quat = quat_mul(quat, offset_quat.expand_as(quat))
        return quat, pos

    def resample(self, container: ModuleContainer, mask: torch.Tensor) -> None:
        """Draw a new goal (pattern) for the envs selected by ``mask``."""
        raise NotImplementedError

    def step(self, container: ModuleContainer, mask: torch.Tensor) -> None:
        """Advance the goals of the envs selected by ``mask`` by one control step."""

    def _sample_uniform(self, container: ModuleContainer, mask: torch.Tensor) -> None:
        """Sample uniform positions and configured orientations for the envs selected by ``mask``."""
        num = int(mask.sum().item())
        device = mask.device
        pos = torch.stack(
            [lo + (hi - lo) * torch.rand(num, device=device) for lo, hi in (self.x_range, self.y_range, self.z_range)],
            dim=-1,
        )
        self._sample_quat(container, mask)
        container.goal_position_in_goal_frame[mask] = pos

    def _sample_quat(self, container: ModuleContainer, mask: torch.Tensor) -> None:
        """Sample orientations for the envs selected by ``mask``."""
        num = int(mask.sum().item())
        if self.goal_orientation == "random":
            quat = random_uniform_quaternion(num, device=mask.device, dtype=torch.float32)
        else:
            quat = torch.tensor(self.goal_orientation, device=mask.device, dtype=torch.float32).expand(num, 4)
        container.goal_orientation_in_goal_frame[mask] = quat

    @staticmethod
    def _control_dt(container: ModuleContainer, name: str) -> float:
        timestep = container.get("timestep")
        frame_skip = container.get("frame_skip")
        if timestep is None or frame_skip is None:
            raise RuntimeError(
                f"goal_pose '{name}' requires 'timestep' and 'frame_skip' in the shared "
                "container. Ensure create_rigid_vsim_envs is listed before it."
            )
        return float(timestep) * float(frame_skip)


class _DiscreteGoal(_GoalType):
    """Static goals, resampled on reset and with probability ``resample_prob`` per env per step."""

    name = "discrete"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.resample_prob = float(config.get("resample_prob", 0.0))

    def resample(self, container: ModuleContainer, mask: torch.Tensor) -> None:
        self._sample_uniform(container, mask)

    def step(self, container: ModuleContainer, mask: torch.Tensor) -> None:
        if self.resample_prob <= 0.0:
            return
        resample = mask & (torch.rand(mask.shape[0], device=mask.device) < self.resample_prob)
        if resample.any():
            self._sample_uniform(container, resample)


class _SinusoidalGoal(_GoalType):
    """Goals moving along a per-env random sinusoid per axis, ``center + amp * sin(2pi f t + phase)``.

    Every reset draws a new pattern: amplitude a random fraction (``min/max_amplitude_fraction``)
    of each axis's half-range, frequency in ``[min_frequency, max_frequency]`` Hz, phase in
    ``[0, 2pi)`` and a direction (time runs forward or backward along the pattern, 50/50), so the
    goal never leaves the configured ranges. Config block ``sinusoidal``.
    """

    name = "sinusoidal"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        sinusoidal_config = config.get("sinusoidal", {})
        self.min_frequency = float(sinusoidal_config.get("min_frequency", 0.2))
        self.max_frequency = float(sinusoidal_config.get("max_frequency", 0.8))
        self.min_amplitude_fraction = float(sinusoidal_config.get("min_amplitude_fraction", 0.3))
        self.max_amplitude_fraction = float(sinusoidal_config.get("max_amplitude_fraction", 1.0))
        if self.min_frequency <= 0.0 or self.min_frequency > self.max_frequency:
            raise ValueError("goal_pose 'sinusoidal' requires 0 < min_frequency <= max_frequency (Hz).")
        if not 0.0 <= self.min_amplitude_fraction <= self.max_amplitude_fraction <= 1.0:
            raise ValueError("goal_pose 'sinusoidal' requires 0 <= min_amplitude_fraction <= max_amplitude_fraction <= 1.")

    def allocate(self, container: ModuleContainer) -> None:
        device = container.device
        n = container.total_num_envs
        ranges = (self.x_range, self.y_range, self.z_range)
        self._center = torch.tensor([(lo + hi) / 2.0 for lo, hi in ranges], device=device, dtype=torch.float32)
        self._half_range = torch.tensor([abs(hi - lo) / 2.0 for lo, hi in ranges], device=device, dtype=torch.float32)
        self._dt = self._control_dt(container, self.name)
        self._time_buf = torch.zeros(n, device=device, dtype=torch.float32)
        self._freq = torch.zeros((n, 3), device=device, dtype=torch.float32)
        self._phase = torch.zeros((n, 3), device=device, dtype=torch.float32)
        self._amp = torch.zeros((n, 3), device=device, dtype=torch.float32)
        self._direction = torch.ones(n, device=device, dtype=torch.float32)  # +1 / -1 per env

    def resample(self, container: ModuleContainer, mask: torch.Tensor) -> None:
        num = int(mask.sum().item())
        device = mask.device
        fraction_span = self.max_amplitude_fraction - self.min_amplitude_fraction
        fraction = self.min_amplitude_fraction + fraction_span * torch.rand((num, 3), device=device)
        self._amp[mask] = fraction * self._half_range
        self._freq[mask] = self.min_frequency + (self.max_frequency - self.min_frequency) * torch.rand((num, 3), device=device)
        self._phase[mask] = 2.0 * torch.pi * torch.rand((num, 3), device=device)
        self._direction[mask] = torch.where(torch.rand(num, device=device) < 0.5, -1.0, 1.0)
        self._time_buf[mask] = 0.0
        self._sample_quat(container, mask)
        container.goal_position_in_goal_frame[mask] = self._positions(mask)

    def step(self, container: ModuleContainer, mask: torch.Tensor) -> None:
        self._time_buf += self._dt
        container.goal_position_in_goal_frame[mask] = self._positions(mask)

    def _positions(self, mask: torch.Tensor) -> torch.Tensor:
        t = (self._direction[mask] * self._time_buf[mask]).unsqueeze(-1)
        return self._center + self._amp[mask] * torch.sin(2.0 * torch.pi * self._freq[mask] * t + self._phase[mask])


class _SquareGoal(_GoalType):
    """Goals moving along the outer edge of a planar square at a per-env random speed.

    The square spans the two axes in ``plane`` (axis indices, default: the two largest
    half-ranges); the third axis stays at the center of its range. Every reset draws a speed in
    ``speed: {min, max}`` (m/s), a random start along the perimeter and a direction (clockwise or
    counterclockwise, 50/50).
    """

    name = "square"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        speed_config = config.get("speed", {})
        self.min_speed = float(speed_config.get("min", 0.01))
        self.max_speed = float(speed_config.get("max", 0.05))
        if self.min_speed <= 0.0 or self.min_speed > self.max_speed:
            raise ValueError("goal_pose 'square' requires 0 < min_speed <= max_speed (m/s).")
        plane = config.get("plane")
        if plane is None:
            half_ranges = [abs(hi - lo) / 2.0 for lo, hi in (self.x_range, self.y_range, self.z_range)]
            order = sorted(range(3), key=lambda i: half_ranges[i], reverse=True)
            self._plane = (order[0], order[1])
        else:
            axes = tuple(int(i) for i in plane)
            if len(axes) != 2 or len(set(axes)) != 2 or not all(0 <= i < 3 for i in axes):
                raise ValueError("goal_pose 'square' config 'plane' must be a list of two distinct axis indices in [0, 2].")
            self._plane = axes

    def allocate(self, container: ModuleContainer) -> None:
        axis0, axis1 = self._plane
        axis2 = ({0, 1, 2} - {axis0, axis1}).pop()
        ranges = (self.x_range, self.y_range, self.z_range)
        lo0, hi0 = sorted(ranges[axis0])
        lo1, hi1 = sorted(ranges[axis1])
        device = container.device
        n = container.total_num_envs
        w, h = hi0 - lo0, hi1 - lo1
        self._axes = (axis0, axis1, axis2)
        self._corners = torch.tensor([[lo0, lo1], [hi0, lo1], [hi0, hi1], [lo0, hi1], [lo0, lo1]], device=device, dtype=torch.float32)
        self._cum_bounds = torch.tensor([0.0, w, w + h, 2 * w + h, 2 * (w + h)], device=device, dtype=torch.float32)
        self._perimeter = float(2 * (w + h))
        self._center_inactive = sum(ranges[axis2]) / 2.0
        self._dt = self._control_dt(container, self.name)
        self._time_buf = torch.zeros(n, device=device, dtype=torch.float32)
        self._speed = torch.zeros(n, device=device, dtype=torch.float32)
        self._start = torch.zeros(n, device=device, dtype=torch.float32)
        self._direction = torch.ones(n, device=device, dtype=torch.float32)  # +1 / -1 per env

    def resample(self, container: ModuleContainer, mask: torch.Tensor) -> None:
        num = int(mask.sum().item())
        device = mask.device
        self._speed[mask] = self.min_speed + (self.max_speed - self.min_speed) * torch.rand(num, device=device)
        self._start[mask] = self._perimeter * torch.rand(num, device=device)
        self._direction[mask] = torch.where(torch.rand(num, device=device) < 0.5, -1.0, 1.0)
        self._time_buf[mask] = 0.0
        self._sample_quat(container, mask)
        container.goal_position_in_goal_frame[mask] = self._positions(self._start[mask])

    def step(self, container: ModuleContainer, mask: torch.Tensor) -> None:
        self._time_buf += self._dt
        distance = self._start[mask] + self._direction[mask] * self._speed[mask] * self._time_buf[mask]
        container.goal_position_in_goal_frame[mask] = self._positions(distance)

    def _positions(self, distance: torch.Tensor) -> torch.Tensor:
        """Map a distance along the perimeter to goal positions ``(N, 3)``."""
        d = distance % self._perimeter
        idx = (d.unsqueeze(-1) >= self._cum_bounds[1:4]).sum(-1)
        start, end = self._corners[idx], self._corners[idx + 1]
        seg_start, seg_end = self._cum_bounds[idx], self._cum_bounds[idx + 1]
        frac = ((d - seg_start) / (seg_end - seg_start + 1e-12)).unsqueeze(-1)
        plane_pos = start + frac * (end - start)

        axis0, axis1, axis2 = self._axes
        out = torch.empty((distance.shape[0], 3), device=distance.device, dtype=torch.float32)
        out[:, axis0] = plane_pos[:, 0]
        out[:, axis1] = plane_pos[:, 1]
        out[:, axis2] = self._center_inactive
        return out


GOAL_TYPES: dict[str, type[_GoalType]] = {cls.name: cls for cls in (_DiscreteGoal, _SinusoidalGoal, _SquareGoal)}


@register_module("goal_pose")
class GoalPoseModule(BaseModule):
    """Samples the goal in its goal frame: ``container.goal_position_in_goal_frame`` ``(N, 3)`` and
    ``container.goal_orientation_in_goal_frame`` ``(N, 4)``, i.e. within the configured
    ``goal_position`` ranges, with no further transform. The goal frame of every env is published as
    ``container.goal_frame_quat`` ``(N, 4)`` / ``goal_frame_pos`` ``(N, 3)`` (its pose in the environment
    frame; identity for types without ``frame``). Observations read the goal-frame values; rewards,
    terminations and visualization convert to the environment frame with :func:`goal_in_env_frame`.

    Every goal type under ``types`` (``discrete``, ``sinusoidal``, ``square``) is used, and the envs
    are split evenly (contiguous blocks, in block order) over them; an optional ``weight``
    (default 1) gives a type a larger share. Goals are resampled for every env selected by the
    reset buffer on every reset call, and the moving types advance every step.

    Config shape (keys other than ``types`` are shared defaults that each type block can override)::

        goal_pose:
          frame: Pinky_Abd
          goal_orientation: [0, 0, 0, 1]
          goal_position: {x: [..], y: [..], z: [..]}
          types:
            square:
              plane: [0, 2]
              speed: {min: 0.001, max: 0.05}
            discrete:
              resample_prob: 0.005
            sinusoidal:
              weight: 2
              sinusoidal: {min_frequency: 0.2, max_frequency: 0.8, min_amplitude_fraction: 0.0, max_amplitude_fraction: 1.0}
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        types = self.config.get("types") or {}
        if not isinstance(types, dict) or not types:
            raise ValueError(f"goal_pose config requires a 'types' mapping with at least one of {sorted(GOAL_TYPES)}.")
        unknown = sorted(set(types) - set(GOAL_TYPES))
        if unknown:
            raise ValueError(f"goal_pose 'types' has unknown goal types {unknown}; expected any of {sorted(GOAL_TYPES)}.")
        blocks = {name: dict(block or {}) for name, block in types.items()}
        shared = {key: value for key, value in self.config.items() if key != "types"}
        # One entry per share: the envs are split evenly over the entries.
        self.entries: list[str] = []
        for name, block in blocks.items():
            weight = block.pop("weight", 1)
            if not isinstance(weight, int) or weight < 1:
                raise ValueError(f"goal_pose '{name}' weight must be a positive integer, got {weight!r}.")
            self.entries += [name] * weight
        self.goal_types = {name: GOAL_TYPES[name]({**shared, **block}) for name, block in blocks.items()}
        self._masks: dict[str, torch.Tensor] | None = None
        self._frames_read = False

    def _allocate(self, container: ModuleContainer) -> None:
        """Allocate the goal buffers, assign each env a type and allocate the type states."""
        n = container.total_num_envs
        device = container.device
        container.goal_position_in_goal_frame = torch.zeros((n, 3), device=device, dtype=torch.float32)
        container.goal_orientation_in_goal_frame = torch.zeros((n, 4), device=device, dtype=torch.float32)
        container.goal_frame_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).repeat(n, 1)
        container.goal_frame_pos = torch.zeros((n, 3), device=device, dtype=torch.float32)

        entry = torch.arange(n, device=device) * len(self.entries) // n
        entry_type = torch.tensor([list(self.goal_types).index(name) for name in self.entries], device=device)
        env_type = entry_type[entry]
        self._masks = {name: env_type == k for k, name in enumerate(self.goal_types)}
        for goal_type in self.goal_types.values():
            goal_type.allocate(container)

    def post_finalize(self, container: ModuleContainer) -> None:
        """Allocate the goal buffers once the container is populated."""
        if "goal_position_in_goal_frame" not in container:
            self._allocate(container)

    def reset(self, container: ModuleContainer) -> None:
        """Resample the goal for the envs selected by the environment's reset buffer."""
        if "goal_position_in_goal_frame" not in container:
            if container.get("total_num_envs") is None or container.get("device") is None:
                raise RuntimeError(
                    "goal_pose requires 'total_num_envs' and 'device' in the shared container. "
                    "Ensure create_rigid_vsim_envs is listed before it."
                )
            self._allocate(container)

        if not self._frames_read:  # once, at the zero pose of the first reset
            self._frames_read = True
            for name, goal_type in self.goal_types.items():
                if goal_type.frame is not None:
                    quat, pos = goal_type.read_frame(container)
                    own = self._masks[name]
                    container.goal_frame_quat[own] = quat[own]
                    container.goal_frame_pos[own] = pos[own]

        reset_buf = container.get("reset_buf")
        for name, goal_type in self.goal_types.items():
            mask = self._masks[name] if reset_buf is None else self._masks[name] & reset_buf
            if mask.any():
                goal_type.resample(container, mask)

    def step(self, container: ModuleContainer) -> None:
        """Advance the goals of every type for its own envs."""
        if self._masks is None:
            return
        for name, goal_type in self.goal_types.items():
            goal_type.step(container, self._masks[name])

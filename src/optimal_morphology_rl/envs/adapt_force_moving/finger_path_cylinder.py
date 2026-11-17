"""The fixed cylinder (the Bota sensor) placed anywhere along the fingertip's curl path, facing the incoming fingertip,
and optionally moved along that path during the episode (a hand-held sensor)."""

from __future__ import annotations

import math
from typing import Any

import torch
import vlearn as v
from vlearn.torch_utils.torch_jit_utils import quat_from_euler_xyz, quat_mul, quat_rotate

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


def _quat_z_to(direction: torch.Tensor) -> torch.Tensor:
    """Quaternions (x, y, z, w) of the shortest rotations taking +z onto the unit vectors ``direction`` (N, 3)."""
    z = torch.zeros_like(direction)
    z[:, 2] = 1.0
    axis = torch.cross(z, direction, dim=1)
    cos = (z * direction).sum(dim=1).clamp(-1.0, 1.0)
    quat = torch.cat([axis, (1.0 + cos).unsqueeze(1)], dim=1)
    # Opposite to +z: 180 deg about x.
    flip = cos < -1.0 + 1e-6
    quat[flip] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=direction.device)
    return quat / quat.norm(dim=1, keepdim=True)


@register_module("finger_path_cylinder")
class FingerPathCylinderModule(BaseModule):
    """Places the fixed cylinder on the fingertip's curl path, top face on the path, axis along it, so the finger
    presses straight down on its top face wherever it is (as on the palm).

    The path: the finger's joints at ``s * path_end`` (joint DOF name -> angle at s = 1; unlisted joints at 0; the
    default is the palm cylinder's contact pose), s in [0, ``path_s_max``]; the fingertip is ``tip_sensor``'s kinematic
    sensor (with its offset). It is tabulated once at start-up (``table_size`` poses, set kinematically on the envs,
    forward kinematics; env frame): tip position, tangent (direction of the tip's motion as s grows) and arc length.

    On reset each env draws s in ``s_range``: the cylinder's top-face center goes to the tip there, moved
    ``contact_offset`` further along the tangent (> 0: the finger must curl past the drawn point to touch), with its +z
    axis against the tangent (the top face faces the incoming fingertip) and its center ``half_height`` behind the top
    face; then a tilt (about the cylinder's own x and y, each in +-``tilt_deg``) and a shift across the axis (each of the
    two perpendicular directions in +-``lateral_range`` m). With ``motion`` the cylinder then moves along the path during
    the episode, every physics substep (list the module in ``pre_gym_step_modules``): the env (unless it drew
    ``static_prob``) holds a target arc position, redrawn in the s range after an exponential hold (``mean_hold_sec``),
    and moves toward it at a speed drawn per target in [0, ``max_speed``] (m/s along the path), e.g. a hand-held
    sensor pushed toward or pulled away from the finger. The values do not follow the curriculum::

        finger_path_cylinder:
          tip_sensor: Middle_DP
          path_end: {Middle_MCP: 1.3116, Middle_PIP: 1.4436, Middle_DIP: 1.4601}
          s_range: [0.3, 1.0]
          contact_offset: 0.0
          half_height: 0.0075
          tilt_deg: 5.0
          lateral_range: 0.002
          motion: {static_prob: 0.3, max_speed: 0.03, mean_hold_sec: 1.5}

    The cylinder's own 'create_objects' pose randomization (height_range / tilt_deg) must be off.
    """

    DEFAULT_PATH_END = {"Middle_MCP": 1.3116, "Middle_PIP": 1.4436, "Middle_DIP": 1.4601}

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        c = self.config
        self.object_name = c.get("object")
        self.tip_sensor = str(c.get("tip_sensor", "Middle_DP"))
        self.path_end = dict(c.get("path_end", self.DEFAULT_PATH_END))
        self.path_s_max = float(c.get("path_s_max", 1.1))
        self.table_size = int(c.get("table_size", 111))
        self.s_range = tuple(float(x) for x in c.get("s_range", [0.3, 1.0]))
        if not 0.0 <= self.s_range[0] <= self.s_range[1] <= self.path_s_max:
            raise ValueError(f"finger_path_cylinder: s_range {self.s_range} must lie in [0, path_s_max {self.path_s_max}].")
        self.contact_offset = float(c.get("contact_offset", 0.0))
        self.half_height = float(c.get("half_height", 0.0075))
        self.tilt = math.radians(float(c.get("tilt_deg", 0.0)))
        self.lateral = float(c.get("lateral_range", 0.0))
        motion = c.get("motion") or {}
        self.static_prob = float(motion.get("static_prob", 1.0))
        self.max_speed = float(motion.get("max_speed", 0.0))
        self.mean_hold = float(motion.get("mean_hold_sec", 1.5))
        self.moving = bool(motion) and self.max_speed > 0.0 and self.static_prob < 1.0

    # ------------------------------------------------------------------ path table
    def _tabulate(self, container: ModuleContainer) -> None:
        """Tip position, tangent and arc length along s (env frame), from kinematic poses of the hand."""
        robot, gym, device = container.robot, container.gym, container.device
        names = list(robot.art_def.get_joint_dof_def_names())
        unknown = sorted(set(self.path_end) - set(names))
        if unknown:
            raise ValueError(f"finger_path_cylinder: path_end joints {unknown} are not hand DOFs ({names}).")
        q_end = torch.tensor([float(self.path_end.get(n, 0.0)) for n in names], device=device)
        sensor = container.kinematic_sensors.get(self.tip_sensor)
        if sensor is None:
            raise RuntimeError(f"finger_path_cylinder: no kinematic sensor '{self.tip_sensor}' ({list(container.kinematic_sensors)}).")
        robot_module = container.get("env").module_manager.get("create_robot")
        n_envs = container.total_num_envs
        s = torch.linspace(0.0, self.path_s_max, self.table_size, device=device)
        tips = torch.zeros((self.table_size, 3), device=device)
        reset_buf = container.reset_buf
        for start in range(0, self.table_size, n_envs):
            rows = s[start:start + n_envs]
            b = len(rows)
            reset_buf[:] = False
            reset_buf[:b] = True
            container.reset_joint_pos_buf[:b] = rows.unsqueeze(1) * q_end
            container.reset_joint_vel_buf[:b] = 0.0
            container.reset_root_transform_buf[:b, :4] = robot_module.fixed_hand_root_quat
            container.reset_root_transform_buf[:b, 4:] = robot_module.fixed_hand_root_pos
            container.reset_root_vel_buf[:b] = 0.0
            gym.set_articulation_kinematic_states(container.gpu_reset_kinematic_state_command_array)
            gym.compute_kinematics()
            sensor.update(gym)
            tips[start:start + b] = sensor.pos_in_world[:b]
        reset_buf[:] = False
        container.reset_joint_pos_buf[:] = 0.0
        tangent = torch.zeros_like(tips)
        tangent[1:-1] = tips[2:] - tips[:-2]
        tangent[0], tangent[-1] = tips[1] - tips[0], tips[-1] - tips[-2]
        tangent = tangent / tangent.norm(dim=1, keepdim=True).clamp(min=1e-9)
        arc = torch.zeros(self.table_size, device=device)
        arc[1:] = torch.cumsum((tips[1:] - tips[:-1]).norm(dim=1), dim=0)
        self.table_s, self.table_tip, self.table_tangent, self.table_arc = s, tips, tangent, arc
        self.arc_range = (float(self._interp(arc, torch.tensor([self.s_range[0]], device=device))[0]),
                          float(self._interp(arc, torch.tensor([self.s_range[1]], device=device))[0]))
        ds = self.path_s_max / (self.table_size - 1)
        at = lambda x: int(round(x / ds))  # noqa: E731
        print(f"FingerPathCylinder: '{self.tip_sensor}' path over s 0..{self.path_s_max:g} x {self.path_end}: arc "
              f"{float(arc[-1]) * 1000:.1f} mm; s_range {list(self.s_range)} = arc {self.arc_range[0] * 1000:.1f}-"
              f"{self.arc_range[1] * 1000:.1f} mm; tip at s=0 {[round(float(x), 4) for x in tips[0]]}, at s=1 "
              f"{[round(float(x), 4) for x in tips[at(1.0)]]} (tangent {[round(float(x), 3) for x in tangent[at(1.0)]]})")

    def _interp(self, values: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """``values`` (table, ...) linearly interpolated at s (N,)."""
        x = (s / self.path_s_max * (self.table_size - 1)).clamp(0.0, self.table_size - 1 - 1e-6)
        i = x.floor().long()
        w = (x - i).reshape(-1, *([1] * (values.dim() - 1)))
        return values[i] * (1.0 - w) + values[i + 1] * w

    def _s_at_arc(self, arc: torch.Tensor) -> torch.Tensor:
        """Inverse of the arc-length table: s at arc lengths (N,)."""
        i = torch.searchsorted(self.table_arc, arc.contiguous()).clamp(1, self.table_size - 1)
        a0, a1 = self.table_arc[i - 1], self.table_arc[i]
        w = ((arc - a0) / (a1 - a0).clamp(min=1e-12)).clamp(0.0, 1.0)
        return self.table_s[i - 1] + w * (self.table_s[i] - self.table_s[i - 1])

    # ------------------------------------------------------------------ module hooks
    def post_finalize(self, container: ModuleContainer) -> None:
        objects = container.get("objects")
        name = self.object_name or container.reward_object_name
        if objects is None or name not in objects:
            raise RuntimeError(f"finger_path_cylinder: object '{name}' is not created by 'create_objects'.")
        obj = objects[name]
        if getattr(obj, "randomized", False):
            raise ValueError("finger_path_cylinder: turn off the cylinder's own height_range / tilt_deg in create_objects.")
        n, device = container.total_num_envs, container.device
        self.trans = torch.zeros((n, 7), device=device)
        self.trans[:, 3] = 1.0
        self.vel = torch.zeros((n, 6), device=device)
        cmd = container.env_group.create_rigid_body_kinematic_state_command(
            v.wrap_gpu_buffer(self.trans), v.wrap_gpu_buffer(self.vel), obj.handle)  # every env (it moves)
        self.cmd = container.gym.create_gpu_array([cmd])
        self._tabulate(container)
        self.arc = torch.zeros(n, device=device)  # current arc position along the path [m]
        self.arc_target = torch.zeros(n, device=device)
        self.speed = torch.zeros(n, device=device)
        self.static = torch.ones(n, dtype=torch.bool, device=device)
        self.tilt_quat = torch.zeros((n, 4), device=device)
        self.tilt_quat[:, 3] = 1.0
        self.shift = torch.zeros((n, 2), device=device)  # across the axis [m]
        self.physics_dt = float(container.timestep)
        self.hold_prob = 1.0 - math.exp(-self.physics_dt / self.mean_hold) if self.moving else 0.0
        container.cylinder_path_s = torch.zeros(n, device=device)  # the cylinder's s (for logging / plots)

    def _draw_arc(self, n: int, device: torch.device) -> torch.Tensor:
        low, high = self.arc_range
        return low + (high - low) * torch.rand(n, device=device)

    def _write(self, container: ModuleContainer) -> None:
        s = self._s_at_arc(self.arc)
        container.cylinder_path_s[:] = s
        tip = self._interp(self.table_tip, s)
        tangent = self._interp(self.table_tangent, s)
        tangent = tangent / tangent.norm(dim=1, keepdim=True)
        quat = quat_mul(_quat_z_to(-tangent), self.tilt_quat)  # tilt about the cylinder's own x / y
        axis = quat_rotate(quat, torch.tensor([[0.0, 0.0, 1.0]], device=s.device).expand_as(tip))
        ex = quat_rotate(quat, torch.tensor([[1.0, 0.0, 0.0]], device=s.device).expand_as(tip))
        ey = quat_rotate(quat, torch.tensor([[0.0, 1.0, 0.0]], device=s.device).expand_as(tip))
        top = tip + self.contact_offset * tangent + self.shift[:, :1] * ex + self.shift[:, 1:] * ey
        self.trans[:, :4] = quat
        self.trans[:, 4:] = top - self.half_height * axis
        container.gym.set_rigid_body_kinematic_states(self.cmd)

    def reset(self, container: ModuleContainer) -> None:
        envs = container.reset_buf
        if not envs.any():
            return
        k, device = int(envs.sum()), container.device
        self.arc[envs] = self._draw_arc(k, device)
        self.arc_target[envs] = self.arc[envs]
        self.speed[envs] = 0.0
        self.static[envs] = torch.rand(k, device=device) < self.static_prob if self.moving else True
        if self.moving:  # the moving ones head for a first target right away
            start = envs & ~self.static
            m = int(start.sum())
            self.arc_target[start] = self._draw_arc(m, device)
            self.speed[start] = self.max_speed * torch.rand(m, device=device)
        roll = (2.0 * torch.rand(k, device=device) - 1.0) * self.tilt
        pitch = (2.0 * torch.rand(k, device=device) - 1.0) * self.tilt
        self.tilt_quat[envs] = quat_from_euler_xyz(roll, pitch, torch.zeros_like(roll))
        self.shift[envs] = (2.0 * torch.rand((k, 2), device=device) - 1.0) * self.lateral
        self._write(container)

    def step(self, container: ModuleContainer) -> None:
        """One physics substep of the motion along the path (nothing to do for static cylinders)."""
        if not self.moving:
            return
        n, device = self.arc.shape[0], self.arc.device
        redraw = ~self.static & (torch.rand(n, device=device) < self.hold_prob)
        if redraw.any():
            k = int(redraw.sum())
            self.arc_target[redraw] = self._draw_arc(k, device)
            self.speed[redraw] = self.max_speed * torch.rand(k, device=device)
        delta = (self.arc_target - self.arc).clamp(-1.0, 1.0)
        step = torch.minimum(delta.abs(), self.speed * self.physics_dt)
        self.arc += torch.where(self.static, 0.0, torch.sign(delta) * step)
        self._write(container)

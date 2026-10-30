"""Reward for spinning the cube about a fixed world axis, in place."""

from __future__ import annotations

from typing import Any

import torch
from vlearn.torch_utils.torch_jit_utils import quat_conjugate, quat_mul

from optimal_morphology_rl.envs.hand_envs.utils import get_reward_object_name
from optimal_morphology_rl.modules.create_kinematic_sensor_module import (
    find_kinematic_sensor,
)
from optimal_morphology_rl.modules.goal_pose_module import goal_in_env_frame
from optimal_morphology_rl.modules.rewards.reward_base_module import RewardBaseModule
from optimal_morphology_rl.modules.rewards.reward_manager_module import register_reward


class OrientationRate:
    """Angular velocity (rad/s, world frame), the mean over the last ``window_steps`` control steps of the per-step
    orientation change.

    Per step the rotation vector of q_t q_{t-1}^-1 divided by dt (quaternions xyzw), not the simulator's angular
    velocity: in contact the latter jitters by several rad/s while the cube stands still and its integral drifts. With
    k > 1 a back and forth vibration cancels and only the net rotation counts. The per-step rotations are averaged
    rather than taking q_t q_{t-k}^-1 once: that wraps beyond pi over the window (a thin pen spun backwards at > pi / (k
    dt) read as a forward roll and was farmed); one step wraps only beyond pi / dt (628 rad/s at 200 Hz). Envs reset
    since the last step restart with no rotation (0 at first, then the rotation since the reset over the window).
    """

    def __init__(self, dt: float, window_steps: int = 1):
        if window_steps < 1:
            raise ValueError(f"OrientationRate window_steps must be >= 1, got {window_steps}.")
        self.dt = dt
        self.window_steps = window_steps
        self.previous: torch.Tensor | None = None  # (num_envs, 4) orientation one step ago
        self.history: torch.Tensor | None = None  # (window_steps, num_envs, 3) ring buffer of per-step rates
        self.index = 0  # slot of the oldest per-step rate
        self.fresh: torch.Tensor | None = None  # envs reset since the last step: no orientation change to measure

    def reset(self, reset_buf: torch.Tensor | None) -> None:
        if self.fresh is not None and reset_buf is not None:
            self.fresh |= reset_buf

    def __call__(self, quat: torch.Tensor) -> torch.Tensor:
        if self.previous is None:
            self.previous = quat.clone()
            self.history = torch.zeros((self.window_steps, quat.shape[0], 3), dtype=quat.dtype, device=quat.device)
            self.fresh = torch.ones(quat.shape[0], dtype=torch.bool, device=quat.device)
        self.previous[self.fresh] = quat[self.fresh]
        self.history[:, self.fresh] = 0.0
        dq = quat_mul(quat, quat_conjugate(self.previous))
        dq = torch.where(dq[:, 3:4] < 0.0, -dq, dq)  # shortest rotation
        sin_half = torch.norm(dq[:, :3], dim=-1, keepdim=True)
        angle = 2.0 * torch.atan2(sin_half, dq[:, 3:4])
        self.history[self.index] = dq[:, :3] * (angle / sin_half.clamp(min=1e-9)) / self.dt
        self.fresh[:] = False
        self.previous[:] = quat
        self.index = (self.index + 1) % self.window_steps
        return self.history.mean(dim=0)


@register_reward("cube_rotation_reward")
class CubeRotationReward(RewardBaseModule):
    """In-hand rotation reward (as in Qi et al., "In-Hand Object Rotation via Rapid Motor Adaptation").

    Per step, with the cube angular velocity w and linear velocity v in the world frame and the unit axis k:
    w is the rotation of the cube's orientation over the step (rotation vector of q_t q_{t-1}^-1, divided by dt), not
    the simulator's angular velocity: in contact the latter jitters by several rad/s while the cube stands still, and
    its integral drifts (a cube tipped over once read as -18 rad in 6 s), which a policy could farm.

        rotation  = clamp(w . k, -max_speed, max_speed) / max_speed      (1 at max_speed rad/s about +k)
        wobble    = |w - (w . k) k|                                     (rad/s about the other axes)
        drift     = |v|                                                 (m/s)

        gate      = exp(-(|p - goal| / position_gate)^2)                (1 without position_gate)
        reward = scale * gate * rotation - wobble_scale * wobble - drift_scale * drift

    The clamp keeps the policy from throwing the cube for one fast spin; the gate, wobble and drift terms keep it
    rotating in place instead of tumbling or rolling away (rolling the cube off the palm also turns it).

    Config::

        cube_rotation_reward:
          axis: [0.0, 0.0, 1.0]   # world axis, normalized
          max_speed: 1.0          # rad/s
          scale: 1.0
          wobble_scale: 0.1
          drift_scale: 1.0
          position_gate: 0.02     # m, optional
    """

    def post_finalize(self, container: Any) -> None:
        self.orientation_rate = OrientationRate(float(container.timestep) * float(container.frame_skip))

    def reset(self, container: Any) -> None:
        self.orientation_rate.reset(container.get("reset_buf"))

    def compute(self, env: Any) -> torch.Tensor | None:
        reward_object_name = get_reward_object_name(env)
        if reward_object_name != "cube":
            return None

        container = env.module_manager.container
        sensor = find_kinematic_sensor(container, reward_object_name)
        if sensor is None:
            raise RuntimeError(
                "cube_rotation_reward: no kinematic sensor for the reward object. "
                "Ensure create_kinematic_sensor is listed."
            )
        axis = torch.tensor(self.config.get("axis", [0.0, 0.0, 1.0]), device=env.device, dtype=torch.float32)
        axis = axis / axis.norm()
        max_speed = float(self.config.get("max_speed", 1.0))
        scale = float(self.config.get("scale", 1.0))
        wobble_scale = float(self.config.get("wobble_scale", 0.1))
        drift_scale = float(self.config.get("drift_scale", 1.0))

        omega = self.orientation_rate(sensor.quat_sensor_to_world)
        speed = omega @ axis
        rotation = speed.clamp(-max_speed, max_speed) / max_speed
        wobble = torch.norm(omega - speed.unsqueeze(-1) * axis, dim=-1)
        drift = torch.norm(sensor.linear_velocity_world, dim=-1)
        position_gate = self.config.get("position_gate")
        if position_gate is not None:
            goal_pos, _ = goal_in_env_frame(container)
            rotation = rotation * torch.exp(-((torch.norm(sensor.pos_in_world - goal_pos, dim=-1) / float(position_gate)) ** 2))

        n = env.total_num_envs
        env.info["rewards"]["rotation_speed_rad_s"] = speed.sum().item() / n
        env.info["rewards"]["rotation_wobble_rad_s"] = wobble.sum().item() / n
        env.info["rewards"]["cube_drift_m_s"] = drift.sum().item() / n
        return scale * rotation - wobble_scale * wobble - drift_scale * drift

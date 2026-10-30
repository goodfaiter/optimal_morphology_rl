"""Reward for tracking a target angular velocity of the cube about a world axis."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.envs.adapt_cube.rewards.cube_rotation_reward import OrientationRate
from optimal_morphology_rl.envs.hand_envs.utils import get_reward_object_name
from optimal_morphology_rl.modules.create_kinematic_sensor_module import (
    find_kinematic_sensor,
)
from optimal_morphology_rl.modules.goal_pose_module import goal_in_env_frame
from optimal_morphology_rl.modules.rewards.reward_base_module import RewardBaseModule
from optimal_morphology_rl.modules.rewards.reward_manager_module import register_reward


@torch.jit.script
def _cube_angular_velocity_reward_jit(
    speed: torch.Tensor,
    target_speed: float,
    normalize: float,
    scale: float,
    capped: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exponential reward based on the deviation from the target angular velocity.

    The target's sign is the spin direction. With ``capped`` the speed in that direction is clamped to
    ``|target_speed|``, so spinning faster than the target still gives 1; without it, overshoot is penalized like a
    shortfall. Returns scaled reward and raw (unscaled) reward for logging.
    """
    target = abs(target_speed)
    speed_along = speed if target_speed >= 0.0 else -speed
    if capped:
        speed_along = torch.clamp(speed_along, max=target)
    raw = torch.exp(-(((speed_along - target) / normalize) ** 2))
    return scale * raw, raw


@register_reward("cube_angular_velocity_reward")
class CubeAngularVelocityReward(RewardBaseModule):
    """Exponential reward for spinning the cube at ``target_speed`` about a world ``axis``.

    ``speed`` is the cube angular velocity about the axis (rad/s, signed), the finite difference of its orientation
    over the control step (see :class:`OrientationRate`; the simulator's angular velocity drifts in contact):

        raw = exp(-((min(speed, target_speed) - target_speed) / normalize)^2)   capped: 1 at or above the target speed
        raw = exp(-((speed - target_speed) / normalize)^2)                      not capped: overshoot is penalized too

    Config::

        cube_angular_velocity_reward:
          axis: [0.0, 1.0, 0.0]   # world axis, normalized (default world y)
          target_speed: 1.0       # rad/s
          normalize: 0.5          # rad/s
          scale: 1.0
          capped: true            # optional (default true): faster than target_speed is not penalized
          rate_window: 0.2        # s, optional (default one control step): speed = net rotation over the window,
                                  # so a vibration does not earn the reward
          position_gate: 0.04     # m, optional: raw *= exp(-(|cube - goal position| / position_gate)^2), so spinning
                                  # the cube out of the hand (or throwing it) pays nothing
    """

    def post_finalize(self, container: Any) -> None:
        dt = float(container.timestep) * float(container.frame_skip)
        window = self.config.get("rate_window")
        self.orientation_rate = OrientationRate(dt, 1 if window is None else max(1, round(float(window) / dt)))

    def reset(self, container: Any) -> None:
        self.orientation_rate.reset(container.get("reset_buf"))

    def compute(self, env: Any) -> torch.Tensor | None:
        reward_object_name = get_reward_object_name(env)  # the cube, or any free object (adapt_ball: the ball)
        container = env.module_manager.container
        sensor = find_kinematic_sensor(container, reward_object_name)
        if sensor is None:
            raise RuntimeError(
                "cube_angular_velocity_reward: no kinematic sensor for the reward object. "
                "Ensure create_kinematic_sensor is listed."
            )
        axis = torch.tensor(self.config.get("axis", [0.0, 1.0, 0.0]), device=env.device, dtype=torch.float32)
        axis = axis / axis.norm()
        target_speed = float(self.config.get("target_speed", 1.0))
        normalize = float(self.config.get("normalize", 0.5))
        scale = float(self.config.get("scale", 1.0))
        capped = bool(self.config.get("capped", True))

        speed = self.orientation_rate(sensor.quat_sensor_to_world) @ axis
        # print(speed)
        # print(speed)
        reward, raw = _cube_angular_velocity_reward_jit(speed, target_speed, normalize, scale, capped)
        position_gate = self.config.get("position_gate")
        if position_gate is not None:
            goal_pos, _ = goal_in_env_frame(container)
            gate = torch.exp(-((torch.norm(sensor.pos_in_world - goal_pos, dim=-1) / float(position_gate)) ** 2))
            reward, raw = reward * gate, raw * gate
        # print("speed", speed, "target_speed", target_speed, "reward", reward)

        env.info["rewards"]["cube_angular_velocity_reward"] = raw.sum().item() / env.total_num_envs
        env.info["rewards"]["cube_angular_speed_rad_s"] = speed.sum().item() / env.total_num_envs
        return reward

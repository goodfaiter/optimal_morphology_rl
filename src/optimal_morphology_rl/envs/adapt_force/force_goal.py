"""Target fingertip force: the goal (module), its observations and the tracking reward."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module
from optimal_morphology_rl.modules.observations.observation_base_module import ObservationBaseModule
from optimal_morphology_rl.modules.observations.observation_manager_module import register_observation
from optimal_morphology_rl.modules.rewards.reward_base_module import RewardBaseModule
from optimal_morphology_rl.modules.rewards.reward_manager_module import register_reward


@register_module("force_goal")
class ForceGoalModule(BaseModule):
    """Per-env target force (N) for a fingertip force sensor, and the measured force.

    Like 'goal_pose', every goal type under ``types`` is used and the envs are split evenly (contiguous blocks, in
    block order) over them; an optional ``weight`` (default 1) gives a type a larger share. Keys other than ``types``
    are shared defaults a type block can override. All targets stay in ``force_range``.

    - ``discrete``: a constant target drawn uniformly from ``force_range`` (0 with probability ``zero_prob``: no contact
      wanted), redrawn on reset and with probability ``resample_prob`` per env per step.
    - ``sinusoidal``: center of the range + amplitude * sin(2 pi f t + phase); every reset draws the frequency in
      ``[min_frequency, max_frequency]`` (Hz), the amplitude as a fraction (``min/max_amplitude_fraction``) of the
      half-range and the phase.

    The measured force is the norm of the force of the force sensor on ``sensor_link`` ('create_force_sensor';
    refreshed by 'update_force_sensors', listed before this). Writes ``container.force_goal`` (N,) and
    ``container.fingertip_force`` (N,), both N::

        force_goal:
          sensor_link: Middle_DP
          force_range: [0.0, 3.0] # N
          types:
            discrete:
              resample_prob: 0.003 # per step (~1.7 s at 200 Hz)
              zero_prob: 0.1
            sinusoidal:
              sinusoidal: {min_frequency: 0.1, max_frequency: 0.5, min_amplitude_fraction: 0.2, max_amplitude_fraction: 1.0}
    """

    TYPES = ("discrete", "sinusoidal")

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.sensor_link = str(self.config.get("sensor_link", "Middle_DP"))
        types = self.config.get("types") or {"discrete": {}}
        unknown = sorted(set(types) - set(self.TYPES))
        if unknown:
            raise ValueError(f"force_goal 'types' has unknown goal types {unknown}; expected any of {list(self.TYPES)}.")
        shared = {key: value for key, value in self.config.items() if key != "types"}
        self.blocks = {name: {**shared, **dict(block or {})} for name, block in types.items()}
        # One entry per share: the envs are split evenly over the entries.
        self.entries = [name for name, block in self.blocks.items() for _ in range(int(block.pop("weight", 1)))]
        low, high = (float(x) for x in self.config.get("force_range", [0.0, 3.0]))
        self.force_range = (low, high)

    def post_finalize(self, container: ModuleContainer) -> None:
        sensors = container.get("force_sensors")
        if sensors is None or self.sensor_link not in sensors.force_sensor_link_names:
            found = None if sensors is None else sensors.force_sensor_link_names
            raise RuntimeError(f"force_goal: no force sensor on '{self.sensor_link}' (create_force_sensor found {found}).")
        self.sensors = sensors
        self.sensor_index = sensors.force_sensor_link_names.index(self.sensor_link)
        n, device = container.total_num_envs, container.device
        self.dt = float(container.timestep) * float(container.frame_skip)
        names = list(self.blocks)
        entry = torch.arange(n, device=device) * len(self.entries) // n
        env_type = torch.tensor([names.index(name) for name in self.entries], device=device)[entry]
        self.masks = {name: env_type == k for k, name in enumerate(names)}
        self.time = torch.zeros(n, device=device)
        self.freq = torch.zeros(n, device=device)
        self.amp = torch.zeros(n, device=device)
        self.phase = torch.zeros(n, device=device)
        container.force_goal = torch.zeros(n, device=device)
        container.fingertip_force = torch.zeros(n, device=device)

    def _discrete(self, container: ModuleContainer, envs: torch.Tensor) -> None:
        n, device = int(envs.sum()), envs.device
        low, high = self.force_range
        goal = low + (high - low) * torch.rand(n, device=device)
        zero = torch.rand(n, device=device) < float(self.blocks["discrete"].get("zero_prob", 0.1))
        container.force_goal[envs] = torch.where(zero, 0.0, goal)

    def _sinusoid_draw(self, envs: torch.Tensor) -> None:
        n, device = int(envs.sum()), envs.device
        cfg = self.blocks["sinusoidal"].get("sinusoidal", {})
        f_lo, f_hi = float(cfg.get("min_frequency", 0.1)), float(cfg.get("max_frequency", 0.5))
        a_lo, a_hi = float(cfg.get("min_amplitude_fraction", 0.2)), float(cfg.get("max_amplitude_fraction", 1.0))
        half = (self.force_range[1] - self.force_range[0]) / 2.0
        self.freq[envs] = f_lo + (f_hi - f_lo) * torch.rand(n, device=device)
        self.amp[envs] = (a_lo + (a_hi - a_lo) * torch.rand(n, device=device)) * half
        self.phase[envs] = 2.0 * torch.pi * torch.rand(n, device=device)
        self.time[envs] = 0.0

    def _sinusoid(self, container: ModuleContainer, envs: torch.Tensor) -> None:
        center = (self.force_range[0] + self.force_range[1]) / 2.0
        wave = torch.sin(2.0 * torch.pi * self.freq[envs] * self.time[envs] + self.phase[envs])
        container.force_goal[envs] = center + self.amp[envs] * wave

    def _measure(self, container: ModuleContainer) -> None:
        container.fingertip_force[:] = torch.norm(self.sensors.force_sensor_buf[:, self.sensor_index, :3], dim=-1)

    def reset(self, container: ModuleContainer) -> None:
        for name, own in self.masks.items():
            envs = own & container.reset_buf
            if not envs.any():
                continue
            if name == "discrete":
                self._discrete(container, envs)
            else:
                self._sinusoid_draw(envs)
                self._sinusoid(container, envs)
        self._measure(container)

    def step(self, container: ModuleContainer) -> None:
        self._measure(container)
        if "discrete" in self.masks:
            prob = float(self.blocks["discrete"].get("resample_prob", 0.0))
            envs = self.masks["discrete"] & (torch.rand_like(self.time) < prob)
            if envs.any():
                self._discrete(container, envs)
        if "sinusoidal" in self.masks:
            self.time += self.dt
            self._sinusoid(container, self.masks["sinusoidal"])


@register_observation("force_goal")
class ForceGoalObservation(ObservationBaseModule):
    """The target force (N) of 'force_goal'."""

    def get_obs_dim(self, env: Any) -> int:
        return 1

    def compute_observation(self, env: Any, out: torch.Tensor) -> None:
        out[:, 0] = env.module_manager.container.force_goal


@register_observation("fingertip_force")
class FingertipForceObservation(ObservationBaseModule):
    """The measured fingertip force (N) of 'force_goal'."""

    def get_obs_dim(self, env: Any) -> int:
        return 1

    def compute_observation(self, env: Any, out: torch.Tensor) -> None:
        out[:, 0] = env.module_manager.container.fingertip_force


@register_reward("force_tracking_reward")
class ForceTrackingReward(RewardBaseModule):
    """``scale * exp(-((force - goal) / normalize)^2)``: the measured fingertip force at the 'force_goal' target. With
    ``max_error`` the error is clipped to +-max_error N first. A list of widths (``normalize: [0.3, 1.0]``) gives the
    mean of their kernels: the wide one still rewards (and gives a gradient toward) a target far off. Logs force_error_n (the clipped |error|, N) and
    force_tracking_reward (the unscaled reward), with fingertip_force_n and force_goal_n::

        force_tracking_reward:
          scale: 1.0
          normalize: 0.2 # N
          max_error: 3.0 # N, optional
    """

    def compute(self, env: Any) -> torch.Tensor | None:
        container = env.module_manager.container
        error = container.fingertip_force - container.force_goal
        normalize = self.config.get("normalize", 0.2)
        widths = [float(w) for w in normalize] if isinstance(normalize, (list, tuple)) else [float(normalize)]
        max_error = self.config.get("max_error")
        clipped = error if max_error is None else error.clamp(-float(max_error), float(max_error))
        reward = sum(torch.exp(-((clipped / w) ** 2)) for w in widths) / len(widths)
        n = env.total_num_envs
        env.info["rewards"]["force_error_n"] = clipped.abs().sum().item() / n
        env.info["rewards"]["force_tracking_reward"] = reward.sum().item() / n
        # env.info["rewards"]["fingertip_force_n"] = container.fingertip_force.sum().item() / n
        # env.info["rewards"]["force_goal_n"] = container.force_goal.sum().item() / n
        return float(self.config.get("scale", 1.0)) * reward


@register_reward("fingertip_guide_reward")
class FingertipGuideReward(RewardBaseModule):
    """``scale * exp(-(distance / normalize)^2)`` from the fingertip to a point on the reward object (a small guide
    toward the contact). With ``only_with_goal`` (default true) only while the 'force_goal' target is above 0; false
    keeps the fingertip at the contact for low / zero targets too (resting on the object, not moving away).

    The fingertip is the kinematic sensor ``tip_sensor`` (default Middle_DP, at the DP tip), the point the reward
    object's kinematic sensor plus ``target_offset`` (m, env frame; e.g. the cylinder's top face center)::

        fingertip_guide_reward:
          scale: 0.1
          normalize: 0.02 # m
          tip_sensor: Middle_DP
          target_offset: [0.0, 0.0, 0.0075] # m
          only_with_goal: true
    """

    def compute(self, env: Any) -> torch.Tensor | None:
        container = env.module_manager.container
        tip = container.kinematic_sensors[str(self.config.get("tip_sensor", "Middle_DP"))]
        target = container.kinematic_sensors[container.reward_object_name]
        offset = torch.tensor(self.config.get("target_offset", [0.0, 0.0, 0.0]), device=env.device, dtype=torch.float32)
        distance = torch.norm(tip.pos_in_world - (target.pos_in_world + offset), dim=-1)
        active = (container.force_goal > 0.0).float() if self.config.get("only_with_goal", True) else 1.0
        raw = torch.exp(-((distance / float(self.config.get("normalize", 0.02))) ** 2)) * active
        env.info["rewards"]["fingertip_guide"] = raw.sum().item() / env.total_num_envs
        env.info["rewards"]["fingertip_target_distance_mm"] = distance.sum().item() / env.total_num_envs * 1000
        return float(self.config.get("scale", 0.1)) * raw


@register_reward("joint_order_reward")
class JointOrderReward(RewardBaseModule):
    """``-scale * sum over pairs of max(0, q_follow - q_lead - margin)``: a joint flexing ahead of the one that should
    lead (rad), e.g. the PIP before the MCP (an arch-like curl: the fingertip comes down on the object from above
    instead of hooking its side). Joints by motor name::

        joint_order_reward:
          scale: 1.0
          margin: 0.2 # rad
          pairs: [[Middle_MCP, Middle_PIP]] # [lead, follow]
    """

    def compute(self, env: Any) -> torch.Tensor | None:
        container = env.module_manager.container
        robot = container.robot
        if not hasattr(self, "_dofs"):  # motor -> joint dof indices of the pairs, on the first call
            names = [robot.art_def.get_motor_def(i).name for i in range(robot.num_motors)]
            pairs = self.config.get("pairs", [["Middle_MCP", "Middle_PIP"]])
            self._dofs = [(robot.motor_to_joint_dof_index[names.index(a)], robot.motor_to_joint_dof_index[names.index(b)]) for a, b in pairs]
        q = robot.get_joint_pos_buf
        margin = float(self.config.get("margin", 0.2))
        excess = sum((q[:, follow] - q[:, lead] - margin).clamp(min=0.0) for lead, follow in self._dofs)
        env.info["rewards"]["joint_order_excess_rad"] = excess.sum().item() / env.total_num_envs
        return -float(self.config.get("scale", 1.0)) * excess

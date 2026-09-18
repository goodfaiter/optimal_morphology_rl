"""Module that runs the tendon_est spring-transformer model to produce tendon forces."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import optimal_morphology_rl_assets
import torch
from tendon_est import ModelRunner

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


def _asset_pkg_root() -> Path:
    return Path(optimal_morphology_rl_assets.__file__).resolve().parent


DEFAULT_MODEL_FILENAME = "2026_09_16_best_spring_transformer_latest.pt"
DEFAULT_MODEL_PATH = _asset_pkg_root() / "models" / "actuators" / DEFAULT_MODEL_FILENAME

# Canonical model input feature keys (checkpoint columns carry a "_data" suffix).
FEATURE_KEYS = (
    "measured_position_rad",
    "desired_position_rad",
    "measured_velocity_rad_per_sec",
)


def _build_tendon_scale_tensor(cfg_value: Any, expected_len: int, device: torch.device, name: str) -> torch.Tensor:
    """Convert a scalar or per-model-tendon list config value to a float tensor."""
    if isinstance(cfg_value, (list, tuple)):
        if len(cfg_value) != expected_len:
            raise RuntimeError(f"TendonEstModule config '{name}' length ({len(cfg_value)}) must match {expected_len}.")
        return torch.tensor(cfg_value, device=device, dtype=torch.float32)
    return torch.full((expected_len,), float(cfg_value), device=device, dtype=torch.float32)


@register_module("tendon_est")
class TendonEstModule(BaseModule):
    """Runs the tendon_est model to produce tendon forces for model-driven tendons.

    A single stateful :class:`~tendon_est.ModelRunner` is loaded and runs ONE
    forward pass per step over ``num_envs x num_model_tendons`` rows (the
    scripted checkpoint keeps its history per row). Per model tendon the model
    inputs are computed from the measured tendon state via the pulley radius
    (``dL = radius [m] * angle [rad]``):

    - ``measured_position_rad``: inverse motor angle from the measured tendon length
    - ``measured_velocity_rad_per_sec``: motor angular rate from the measured tendon speed
    - ``desired_position_rad``: measured position plus the scaled policy action,
      clamped to ``[desired_min, desired_max]``

    ``pully_radius`` and ``zero_offset_length`` apply to the model-driven
    tendons only (scalar, or one entry per ``model_tendon_indices`` entry) —
    tendons excluded from the model (e.g. a rigid DIP) need no entries.

    The predicted tendon force is written to ``container.tendon_force_buf`` for
    the model-driven tendon columns (see ``container.tendon_model_indices``)
    so ``robot_control_tendons`` can apply it via ``set_spatial_tendon_forces``.
    Tendon columns not listed in ``model_tendon_indices`` (e.g. the DIP tendon)
    are left untouched and remain under ``robot_control_tendons``'s own control.
    """

    def finalize(self, container: ModuleContainer) -> None:
        """Validate cross-module dependencies and the model-driven tendon config."""
        if container.get("robot") is None:
            raise RuntimeError(
                "TendonEstModule requires 'robot' in the shared container. "
                "Ensure the 'create_robot' module is listed before 'tendon_est'."
            )
        if container.get("env") is None:
            raise RuntimeError("TendonEstModule requires 'env' in the shared container.")

        robot = container.robot
        if not robot.use_tendon:
            raise RuntimeError(
                "TendonEstModule only supports tendon-driven robot configurations "
                "(the 'create_robot' config must set 'use_tendon: true')."
            )

        indices = self.config.get("model_tendon_indices")
        if not isinstance(indices, (list, tuple)) or len(indices) == 0:
            raise RuntimeError("TendonEstModule config missing 'model_tendon_indices': the list of tendon columns the model drives.")
        indices = list(indices)
        for idx in indices:
            if not isinstance(idx, int) or not (0 <= idx < robot.num_tendons):
                raise RuntimeError(
                    f"TendonEstModule config 'model_tendon_indices' entries must be tendon indices in [0, {robot.num_tendons}), got {idx}."
                )
        self.model_tendon_indices = indices

    def post_finalize(self, container: ModuleContainer) -> None:
        """Load the model runner and allocate the force buffer."""
        robot = container.robot
        device = container.device
        num_model_tendons = len(self.model_tendon_indices)

        model_path = Path(self.config.get("model_path") or DEFAULT_MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(
                f"TendonEstModule model not found at '{model_path}'. Provide the 'tendon_est' "
                "'model_path' config or update the default model path."
            )
        # One forward pass per step over num_envs x num_model_tendons rows.
        self.model = ModelRunner(str(model_path), device=str(device), num_envs=container.total_num_envs * num_model_tendons)

        force_idx = next((i for i, column in enumerate(self.model.output_columns) if "force" in column), None)
        if force_idx is None:
            raise RuntimeError(f"TendonEstModule: no force output column in the model outputs ({self.model.output_columns}).")
        self.force_output_idx = force_idx

        order: list[str] = []
        for column in self.model.input_columns:
            key = column.removesuffix("_data")
            if key not in FEATURE_KEYS:
                raise RuntimeError(f"TendonEstModule: unsupported model input column '{column}'.")
            order.append(key)
        self._feature_order = order

        container.tendon_force_buf = torch.zeros((container.total_num_envs, robot.num_tendons), device=device, dtype=torch.float32)
        container.tendon_model_indices = torch.tensor(self.model_tendon_indices, device=device, dtype=torch.long)

        self.pully_radius = _build_tendon_scale_tensor(self.config.get("pully_radius", 0.011), num_model_tendons, device, "pully_radius")
        zero_offset_length = self.config.get("zero_offset_length")
        if zero_offset_length is None:
            raise RuntimeError(
                "TendonEstModule config missing 'zero_offset_length': a per-model-tendon list "
                "of tendon zero-offset lengths (rigid tendons need no entry)."
            )
        self.zero_offset_length = _build_tendon_scale_tensor(zero_offset_length, num_model_tendons, device, "zero_offset_length")
        self.desired_min = float(self.config.get("desired_min", 0.0))
        self.desired_max = float(self.config.get("desired_max", 2.0 * math.pi))

    def step(self, container: ModuleContainer) -> None:
        """Feed the measured/desired joint state of every model tendon in ONE forward pass."""
        robot = container.robot
        num_model_tendons = len(self.model_tendon_indices)

        tendon_lengths = robot.get_tendon_lengths_buf[:, self.model_tendon_indices]
        tendon_vels = robot.get_tendon_vel_buf[:, self.model_tendon_indices]
        actions = container.scaled_act_buf[:, container.active_dof_slice]

        # Map each model tendon to its policy-action slot within the
        # policy-controlled tendon columns (fixed tendons have no action).
        dof_indices = [int(v) for v in container.active_dof_indices.tolist()]
        action_slots = [dof_indices.index(tendon_idx) for tendon_idx in self.model_tendon_indices]

        # dL = radius [m] * angle [rad] (model tendons only)
        measured_pos_rad = -1.0 * (tendon_lengths - self.zero_offset_length) / self.pully_radius
        measured_vel_rad_per_sec = -1.0 * tendon_vels / self.pully_radius
        desired_pos_rad = torch.clamp(
            measured_pos_rad + actions[:, action_slots],
            min=self.desired_min,
            max=self.desired_max,
        )

        feature_map = {
            "measured_position_rad": measured_pos_rad,
            "desired_position_rad": desired_pos_rad,
            "measured_velocity_rad_per_sec": measured_vel_rad_per_sec,
        }
        raw = torch.stack([feature_map[key] for key in self._feature_order], dim=-1)  # [num_envs, T, 3]

        # Env-major flattening: row = env * num_model_tendons + tendon.
        timestep = raw.reshape(container.total_num_envs * num_model_tendons, 3)
        output = self.model.forward(timestep)

        force = -1.0 * output[:, 0, self.force_output_idx]
        force = force.reshape(container.total_num_envs, num_model_tendons)
        container.tendon_force_buf[:, self.model_tendon_indices] = force

    def reset(self, container: ModuleContainer) -> None:
        """Reset the internal model state of the environments selected by the reset buffer."""
        num_model_tendons = len(self.model_tendon_indices)
        # Env i owns the tendon rows [i * T, (i + 1) * T) (env-major flattening).
        self.model.reset_idx(container.reset_buf.repeat_interleave(num_model_tendons))

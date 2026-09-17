"""Module that computes stretch-restoring forces for rigid tendons."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


def _cfg_floats(cfg_value: Any, expected_len: int, name: str) -> list[float]:
    """Convert a scalar or list config value to a list of the expected length."""
    if isinstance(cfg_value, (list, tuple)):
        if len(cfg_value) != expected_len:
            raise RuntimeError(f"RigidTendons config '{name}' length ({len(cfg_value)}) must match {expected_len}.")
        return [float(value) for value in cfg_value]
    return [float(cfg_value)] * expected_len


@register_module("rigid_tendons")
class RigidTendons(BaseModule):
    """Computes stretch-restoring forces for the rigid tendon columns.

    The rigid tendons are derived from the policy mask built by the
    robot_control modules: every tendon excluded from the policy actions
    (``active_dof_mask == False``, e.g. the DIP tendon for
    ``fixed_tendon_substrings``) is rigid. For each of them the module applies
    a PD-style stretch-restoring force toward its rest length; the force is
    only applied when the tendon is longer than its rest length:

    ``force = clamp(length - rest_length, 0) * stretch_kp - velocity * stretch_kd``

    The computed force is written to ``container.rigid_tendon_force_buf``
    (rigid tendon columns listed in ``container.rigid_tendon_indices``) so
    ``robot_control_tendons`` can add it to the tendon controls before the
    forces are applied via ``set_spatial_tendon_forces``.
    """

    def finalize(self, container: ModuleContainer) -> None:
        """Validate cross-module dependencies."""
        if container.get("robot") is None:
            raise RuntimeError(
                "RigidTendons requires 'robot' in the shared container. "
                "Ensure the 'create_robot' module is listed before 'rigid_tendons'."
            )
        if container.get("env") is None:
            raise RuntimeError("RigidTendons requires 'env' in the shared container.")

        robot = container.robot
        if not robot.use_tendon:
            raise RuntimeError(
                "RigidTendons only supports tendon-driven robot configurations " "(the 'create_robot' config must set 'use_tendon: true')."
            )

    def post_finalize(self, container: ModuleContainer) -> None:
        """Derive the rigid tendon columns and allocate the force buffer."""
        robot = container.robot
        device = container.device

        if container.get("active_dof_mask") is None:
            raise RuntimeError(
                "RigidTendons requires 'active_dof_mask' in the shared container. "
                "Ensure one of the robot_control_* modules is listed so the policy mask is built."
            )

        rigid_indices = torch.nonzero(~container.active_dof_mask, as_tuple=False).flatten()
        container.rigid_tendon_indices = rigid_indices.to(device=device, dtype=torch.long)

        num_rigid_tendons = len(rigid_indices)
        self.rest_lengths = _cfg_floats(self.config.get("rest_length", 0.0665), num_rigid_tendons, "rest_length")
        self.stretch_kps = _cfg_floats(self.config.get("stretch_kp", 10000.0), num_rigid_tendons, "stretch_kp")
        self.stretch_kds = _cfg_floats(self.config.get("stretch_kd", 0.0), num_rigid_tendons, "stretch_kd")
        self.force_min = float(self.config.get("force_min", 0.0))

        container.rigid_tendon_force_buf = torch.zeros((container.total_num_envs, robot.num_tendons), device=device, dtype=torch.float32)

    def step(self, container: ModuleContainer) -> None:
        """Compute the stretch-restoring force for each rigid tendon."""
        robot = container.robot
        buf = container.rigid_tendon_force_buf

        lengths = robot.get_tendon_lengths_buf
        vels = robot.get_tendon_vel_buf

        for column, rest, kp, kd in zip(container.rigid_tendon_indices.tolist(), self.rest_lengths, self.stretch_kps, self.stretch_kds):
            stretch = torch.clamp(lengths[:, column] - rest, min=0.0)
            force = stretch * kp - vels[:, column] * kd
            buf[:, column] = torch.clamp(force, min=self.force_min)

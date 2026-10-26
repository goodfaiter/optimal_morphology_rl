"""Module that randomizes the servo spools (pulley radius and slider armature) of 'create_spool'."""

from __future__ import annotations

from typing import Any

import torch
import vlearn as v

from optimal_morphology_rl.modules.base_module import BaseModule, curriculum_range
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


def _optional_range(config: dict[str, Any], key: str) -> tuple[float, float] | None:
    """Validate an optional positive ``[min, max]`` range."""
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise RuntimeError(f"RandomizeSpoolModule config '{key}' must be a [min, max] list, got {value}.")
    low, high = float(value[0]), float(value[1])
    if not 0.0 < low <= high:
        raise RuntimeError(f"RandomizeSpoolModule config '{key}' must satisfy 0 < min <= max, got {value}.")
    return low, high


@register_module("randomize_spool")
class RandomizeSpoolModule(BaseModule):
    """Randomizes the pulley radius and the slider armature of the 'create_spool' spools.

    Both are resampled uniformly on reset, per tendon:

    - ``pully_radius_range`` [m]: per env. Only the spool model uses the radius (theta = q / r,
      F = tau / r, tau_e = -r T), so it is set on ``container.spool`` directly.
    - ``armature_range`` [kg]: the slider DOF armature, i.e. the reflected rotor inertia J / r**2.
      It is a property of the shared articulation definition, so there is one value per env set
      (per env with ``create_rigid_vsim_envs.num_envs_per_set: 1``); an env set is resampled
      when any of its envs resets, and the value is written to the simulator.

    Afterwards the spool inertia J_eff = (armature + slider_mass) * r**2 (used by the force law's
    Karnopp stop torque) is recomputed, so it matches what the simulator integrates. At least one
    range is required. Both ranges follow the training curriculum from the spool's configured values (at 0) to
    the full range (at 1).

    Config shape::

        randomize_spool:
          pully_radius_range: [0.0045, 0.0055]
          armature_range: [38.4, 57.6]
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.radius_range = _optional_range(self.config, "pully_radius_range")
        self.armature_range = _optional_range(self.config, "armature_range")
        if self.radius_range is None and self.armature_range is None:
            raise RuntimeError("RandomizeSpoolModule needs 'pully_radius_range' and/or 'armature_range'.")
        self.spool = None

    def _setup(self, container: ModuleContainer) -> None:
        """Bind the spool and build the armature command (after all post_finalize hooks)."""
        self.spool = container.get("spool")
        if self.spool is None:
            raise RuntimeError("RandomizeSpoolModule requires the 'create_spool' module.")
        self.nominal_radius = self.spool.pully_radius.clone()
        self.nominal_armature = self.spool.slider_armature[0].clone()  # per tendon (the same in every env)
        if self.armature_range is None:
            return

        robot = container.robot
        device = container.device
        num_env_sets = len(container.num_envs)
        # Env set of every env (envs are numbered set by set).
        self.env_set_index = torch.repeat_interleave(
            torch.arange(num_env_sets, device=device), torch.tensor(container.num_envs, device=device)
        )
        # The command writes every DOF, so the other DOFs are filled with their defined armature.
        num_dofs = robot.num_joints
        defined = torch.tensor([robot.art_def.get_joint_dof_def(i).armature for i in range(num_dofs)], device=device, dtype=torch.float32)
        container.set_dof_armature_buf = defined.repeat(num_env_sets, 1)
        command = container.env_group.create_joint_dof_property_command(
            v.JointDofProperty.ARMATURE,
            v.wrap_gpu_buffer(container.set_dof_armature_buf),
            robot.def_handle,
            (0, num_dofs),
        )
        container.gpu_set_dof_armature_cmd = container.gym.create_gpu_array([command])

    def reset(self, container: ModuleContainer) -> None:
        """Resample the spool parameters of the resetting envs and update the servo inertia."""
        if self.spool is None:
            self._setup(container)
        reset_buf = container.reset_buf
        if not reset_buf.any():
            return
        spool = self.spool

        if self.radius_range is not None:
            low, high = curriculum_range(*self.radius_range, self.curriculum_factor, nominal=self.nominal_radius)
            radius = low + (high - low) * torch.rand_like(spool.pully_radius)
            spool.pully_radius[reset_buf] = radius[reset_buf]

        if self.armature_range is not None:
            buf = container.set_dof_armature_buf
            set_mask = torch.zeros(buf.shape[0], dtype=torch.bool, device=buf.device)
            set_mask[self.env_set_index[reset_buf]] = True
            low, high = curriculum_range(*self.armature_range, self.curriculum_factor, nominal=self.nominal_armature)
            samples = low + (high - low) * torch.rand((buf.shape[0], spool.dof_indices.numel()), device=buf.device)
            dofs = spool.dof_indices
            buf[:, dofs] = torch.where(set_mask.unsqueeze(1), samples, buf[:, dofs])
            container.gym.set_joint_dof_properties(container.gpu_set_dof_armature_cmd)
            spool.slider_armature[:] = buf[self.env_set_index][:, dofs]

        spool.update_inertia()

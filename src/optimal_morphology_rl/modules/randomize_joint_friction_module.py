"""Module that randomizes and applies the articulation joint friction."""

from __future__ import annotations

from typing import Any

import torch
import vlearn as v

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


def _validate_range(cfg_value: Any) -> tuple[float, float]:
    """Validate a domain-randomization range and return it as floats."""
    if not isinstance(cfg_value, (list, tuple)) or len(cfg_value) != 2:
        raise RuntimeError(f"RandomizeJointFrictionModule config 'joint_friction_range' must be a [min, max] list, got {cfg_value}.")
    friction_min, friction_max = float(cfg_value[0]), float(cfg_value[1])
    if friction_min > friction_max:
        raise RuntimeError(
            f"RandomizeJointFrictionModule config 'joint_friction_range' min ({friction_min}) must be <= max ({friction_max})."
        )
    return friction_min, friction_max


@register_module("randomize_joint_friction")
class RandomizeJointFrictionModule(BaseModule):
    """Randomizes and applies the articulation joint friction.

    When a ``joint_friction_range`` config (``[min, max]``) is given, a
    joint friction value is sampled uniformly from that range at every reset
    (when at least one env resets) and applied via ``gym.set_joint_properties``
    on the ``JointProperty.JOINT_FRICTION`` property. Only the revolute DOFs
    are randomized; the other DOFs (e.g. the prismatic tendon sliders) keep
    the friction of their vsim joint definition.

    Note: joint properties belong to the shared articulation definition, so
    one value applies to every env of an env set (see
    ``create_joint_property_command``) — this is a global randomization, not
    per-env.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.friction_min: float | None = None
        self.friction_max: float | None = None

    def finalize(self, container: ModuleContainer) -> None:
        """Validate that the robot is available."""
        if container.get("robot") is None:
            raise RuntimeError(
                "RandomizeJointFrictionModule requires 'robot' in the shared container. "
                "Ensure the 'create_robot' module is listed before 'randomize_joint_friction'."
            )

    def post_finalize(self, container: ModuleContainer) -> None:
        """Allocate the friction buffer and create the GPU command."""
        if container.get("reset_buf") is None:
            raise RuntimeError(
                "RandomizeJointFrictionModule requires 'reset_buf' in the shared container. "
                "Ensure 'termination' is loaded."
            )
        robot = container.robot
        device = container.device

        joint_friction_range = self.config.get("joint_friction_range")
        if joint_friction_range is None:
            raise RuntimeError(
                "RandomizeJointFrictionModule config missing 'joint_friction_range': a [min, max] list of joint friction values."
            )
        self.friction_min, self.friction_max = _validate_range(joint_friction_range)

        num_dofs = robot.num_joints
        # Joint friction is a property of the shared articulation def: one row
        # per env set (all envs in a set share it).
        num_env_sets = len(container.num_envs)
        self.revolute_dof_mask = robot.revolute_dof_mask
        # The command writes every DOF, so the non-revolute ones are filled with their defined friction.
        defined_friction = torch.zeros(num_dofs, device=device, dtype=torch.float32)
        for i in range(robot.art_def.get_num_joint_defs()):
            joint_def = robot.art_def.get_joint_def(i)
            start = joint_def.dof_start_index
            defined_friction[start : start + joint_def.dof] = joint_def.joint_friction
        container.set_joint_friction_buf = defined_friction.repeat(num_env_sets, 1)
        self._randomize(container)

        command = container.env_group.create_joint_property_command(
            v.JointProperty.JOINT_FRICTION,
            v.wrap_gpu_buffer(container.set_joint_friction_buf),
            robot.def_handle,
            (0, num_dofs),
        )
        container.gpu_set_joint_friction_cmd = container.gym.create_gpu_array([command])
        print(
            f"RandomizeJointFrictionModule: joint_friction_range = [{self.friction_min}, {self.friction_max}] "
            f"over revolute dofs {torch.nonzero(self.revolute_dof_mask).flatten().tolist()}"
        )

    def _randomize(self, container: ModuleContainer) -> None:
        """Sample the revolute DOF frictions uniformly from the configured range."""
        buf = container.set_joint_friction_buf
        samples = torch.rand(buf.shape, device=buf.device) * (self.friction_max - self.friction_min) + self.friction_min
        buf[:] = torch.where(self.revolute_dof_mask, samples, buf)

    def reset(self, container: ModuleContainer) -> None:
        """Randomize and apply the joint friction when at least one env resets."""
        if container.reset_buf.sum().item() == 0:
            return
        self._randomize(container)
        container.gym.set_joint_properties(container.gpu_set_joint_friction_cmd)

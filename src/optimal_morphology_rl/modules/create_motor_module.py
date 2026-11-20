"""Servo motors directly on the hand's revolute joints (no tendons): mapping, parameters and state buffers."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module
from optimal_morphology_rl.modules.robot_control.helpers import build_action_space


@register_module("create_motor")
class CreateMotorModule(BaseModule):
    """Owns the joint servos of a tendon-free hand (``container.motor``): one per policy-controlled revolute joint
    motor, with the motor angle = the joint angle. 'update_motor' reads the state every physics step:

        theta = q,  omega = qdot              (qdot: the finite difference of q over one physics step)

    The servo inertia is the joint armature (urdf_to_vsim.py full_motor: J on every joint), ``j_eff`` per motor.

    The policy-controlled motors are the 'create_robot' motors not matching its ``passive_motor_substrings`` (default
    ``[abd]``; ``[]``: every motor), or those matching ``active_motor_substrings``, in the vsim's motor order. The
    action is the change of the motor angle (rad). ``hold`` says how it is held over the physics substeps of a control
    step: ``offset`` (default: desired = current angle + action every substep) or ``target`` (with
    'update_motor_target': desired fixed once per control step). ``create_robot.use_tendon`` must be false.

    Lifecycle: ``create_motor`` in ``init_modules`` (builds the action space), 'update_motor_target' in
    ``pre_physics_step_modules`` after 'process_variable_actions', 'update_motor' then 'apply_motor_servo' in
    ``pre_gym_step_modules``.

    Config shape::

        create_motor: {}
    """

    def finalize(self, container: ModuleContainer) -> None:
        if container.get("robot") is None:
            raise RuntimeError("create_motor requires 'create_robot' to be listed before it.")
        robot = container.robot
        if robot.use_tendon:
            raise RuntimeError("create_motor drives the joints directly: set 'create_robot.use_tendon: false'.")
        build_action_space(container)

        art_def = robot.art_def
        motors = [int(i) for i in container.active_dof_indices.tolist()]  # policy-controlled motors, in order
        if not motors:
            raise RuntimeError("create_motor: no policy-controlled motor (check passive_motor_substrings).")
        self.motor_indices_list = motors
        self.dof_indices_list = [int(art_def.get_motor_def(i).dof_index) for i in motors]
        self.names = [art_def.get_motor_def(i).name for i in motors]
        not_revolute = [n for n, d in zip(self.names, self.dof_indices_list) if not bool(robot.revolute_dof_mask[d])]
        if not_revolute:
            raise RuntimeError(f"create_motor: motors {not_revolute} are not on revolute joints.")
        self.armatures = [float(art_def.get_joint_dof_def(d).armature) for d in self.dof_indices_list]
        self.num_motors = len(motors)
        self.hold = "offset"  # 'update_motor_target' sets "target" in its finalize
        print(f"CreateMotor: {self.num_motors} joint servos {self.names}, armature {self.armatures} kg m^2")
        container.motor = self

    def post_finalize(self, container: ModuleContainer) -> None:
        device = container.device
        num_envs = container.total_num_envs
        physics_dt = container.get("physics_dt")
        if physics_dt is None:
            raise RuntimeError("create_motor requires 'physics_dt' ('create_rigid_vsim_envs' listed before it).")
        self.dt = float(physics_dt)

        self.motor_indices = torch.tensor(self.motor_indices_list, device=device, dtype=torch.long)
        self.dof_indices = torch.tensor(self.dof_indices_list, device=device, dtype=torch.long)
        self.j_eff = torch.tensor(self.armatures, device=device, dtype=torch.float32).expand(num_envs, self.num_motors).clone()
        if torch.any(self.j_eff <= 0.0):
            raise RuntimeError("create_motor: the motor joints need a positive armature (the servo inertia).")

        shape = (num_envs, self.num_motors)
        self.theta = torch.zeros(shape, device=device, dtype=torch.float32)
        self.omega = torch.zeros(shape, device=device, dtype=torch.float32)
        self.last_pos = torch.zeros(shape, device=device, dtype=torch.float32)
        self.target = torch.zeros(shape, device=device, dtype=torch.float32)
        self.just_reset = torch.ones(num_envs, device=device, dtype=torch.bool)

    def export_info(self, container: ModuleContainer) -> dict[str, Any]:
        """The motors the actions drive, in order, and how a held action is applied between policy steps."""
        return {"motor": {"names": list(self.names), "hold": self.hold}}

    def action_delta_angle(self, container: ModuleContainer) -> torch.Tensor:
        """The policy's scaled action per motor (rad), ``(num_envs, num_motors)``."""
        return container.scaled_act_buf[:, container.active_dof_slice]

    def commanded_delta_angle(self, container: ModuleContainer) -> torch.Tensor:
        """The commanded change of the motor angle now (desired = theta + this): the action (``hold: offset``) or the
        held target less the current angle (``hold: target``; envs reset since the last read: their action)."""
        if self.hold == "target":
            return torch.where(self.just_reset.unsqueeze(-1), self.action_delta_angle(container), self.target - self.theta)
        return self.action_delta_angle(container)

    def joint_velocity(self, container: ModuleContainer) -> torch.Tensor:
        """Motor joint velocity (rad/s): the finite difference of the joint positions since the last physics step (the
        simulator's velocity reads an offset while a loaded joint stands still); the reset velocity after a reset."""
        pos = container.robot.get_joint_pos_buf[:, self.dof_indices]
        velocity = (pos - self.last_pos) / self.dt
        return torch.where(self.just_reset.unsqueeze(-1), container.reset_joint_vel_buf[:, self.dof_indices], velocity)

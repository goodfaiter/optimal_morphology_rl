"""Servo spools on the tendon actuator sliders: mapping, parameters and per-step state."""

from __future__ import annotations

import torch

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module
from optimal_morphology_rl.modules.robot_control.helpers import build_action_space
from optimal_morphology_rl.modules.tendon_scale_utils import build_tendon_scale_tensor


@register_module("create_spool")
class CreateSpoolModule(BaseModule):
    """Owns the servo spools that wind the tendons, one per model tendon (``container.spool``).

    The vsim generator (urdf_to_vsim.py) prepends a motorized prismatic slider to every actuated
    tendon, with +q pulling the tendon. The slider is the rim of the spool, so the spool state is
    read straight from the simulator:

        theta_m = q / r,  omega_m = qdot / r            (winding positive; qdot from 'update_robot')
        tau_e   = -r * T                                 (T: measured tendon tension, previous step)

    The servo inertia is the slider's armature: the simulator integrates
    J_eff = (armature + slider_mass) * r**2, so the armature should be the reflected rotor inertia
    J / r**2. ``pully_radius``, ``slider_armature`` and ``j_eff`` are per env and tendon
    ``(num_envs, num_model_tendons)`` so they can be randomized ('randomize_spool'); after
    changing the radius or armature call :meth:`update_inertia`.

    Lifecycle: list it in ``init_modules`` (it builds the tendon action space, replacing
    'robot_control_tendons') and in ``pre_physics_step_modules`` before the force law that reads
    its state (e.g. 'apply_m5_forces'). On the step after a reset the state uses the commanded
    reset joint state and zero tension.

    Config shape::

        create_spool:
          pully_radius: 0.005             # m, scalar or per model tendon
          model_tendon_indices: [0, 1]    # default: every policy-controlled tendon
          slider_joint_suffix: _slider_joint
    """

    def finalize(self, container: ModuleContainer) -> None:
        if container.get("robot") is None:
            raise RuntimeError(
                "CreateSpool requires 'robot' in the shared container. Ensure the 'create_robot' module is listed before 'create_spool'."
            )
        if container.get("env") is None:
            raise RuntimeError("CreateSpool requires 'env' in the shared container.")

        robot = container.robot
        if not robot.use_tendon:
            raise RuntimeError(
                "CreateSpool only supports tendon-driven robot configurations (the 'create_robot' config must set 'use_tendon: true')."
            )
        build_action_space(container)

        # Default: every policy-controlled tendon is wound by a spool.
        active = [int(i) for i in container.active_dof_indices.tolist()]
        indices = self.config.get("model_tendon_indices", active)
        if not isinstance(indices, (list, tuple)) or len(indices) == 0:
            raise RuntimeError("CreateSpool config 'model_tendon_indices' must be a non-empty list of tendon indices.")
        for idx in indices:
            if idx not in active:
                raise RuntimeError(f"CreateSpool 'model_tendon_indices' entry {idx} is not a policy-controlled tendon {active}.")
        self.model_tendon_indices = list(indices)
        self.action_slots_list = [active.index(idx) for idx in self.model_tendon_indices]

        # Map each tendon to its slider motor and DOF by the generator's naming convention.
        suffix = str(self.config.get("slider_joint_suffix", "_slider_joint"))
        art_def = robot.art_def
        motor_by_joint = {art_def.get_motor_def(i).joint_name: i for i in range(robot.num_motors)}
        link_mass = {art_def.get_link_def(i).name: art_def.get_link_def(i).mass for i in range(robot.num_links)}
        self.motor_indices_list = []
        self.dof_indices_list = []
        self.slider_armatures = []  # vsim armature of the slider DOF, kg
        self.slider_masses = []  # slider link mass, kg
        for idx in self.model_tendon_indices:
            tendon_name = art_def.get_spatial_tendon_def(idx).name
            joint_name = tendon_name + suffix
            if joint_name not in motor_by_joint:
                raise RuntimeError(
                    f"CreateSpool: no motor on slider joint '{joint_name}' for tendon '{tendon_name}'. "
                    "Regenerate the .vsim with ADD_TENDON_SLIDERS = True."
                )
            motor_idx = motor_by_joint[joint_name]
            dof_idx = int(art_def.get_motor_def(motor_idx).dof_index)
            self.motor_indices_list.append(motor_idx)
            self.dof_indices_list.append(dof_idx)
            self.slider_armatures.append(float(art_def.get_joint_dof_def(dof_idx).armature))
            self.slider_masses.append(float(link_mass.get(tendon_name + suffix.removesuffix("_joint"), 0.0)))
        container.spool = self

    @property
    def slider_inertias(self) -> list[float]:
        """Defined armature + slider mass per model tendon, kg."""
        return [a + m for a, m in zip(self.slider_armatures, self.slider_masses)]

    def post_finalize(self, container: ModuleContainer) -> None:
        device = container.device
        num_envs = container.total_num_envs
        num_tendons = len(self.model_tendon_indices)

        timestep = container.get("timestep")
        frame_skip = container.get("frame_skip")
        if timestep is None or frame_skip is None:
            raise RuntimeError(
                "CreateSpool requires 'timestep' and 'frame_skip' in the shared container. "
                "Ensure the 'create_rigid_vsim_envs' module is listed before 'create_spool'."
            )
        self.dt = float(timestep) * float(frame_skip)

        radius = build_tendon_scale_tensor(self.config.get("pully_radius", 0.011), num_tendons, device, "pully_radius")
        if torch.any(radius <= 0.0):
            raise RuntimeError("CreateSpool config 'pully_radius' entries must be positive.")
        self.pully_radius = radius.expand(num_envs, num_tendons).clone()

        self.action_slots = torch.tensor(self.action_slots_list, device=device, dtype=torch.long)
        self.motor_indices = torch.tensor(self.motor_indices_list, device=device, dtype=torch.long)
        self.dof_indices = torch.tensor(self.dof_indices_list, device=device, dtype=torch.long)

        # Rotary inertia the simulator integrates (per env and tendon).
        self.slider_mass = torch.tensor(self.slider_masses, device=device, dtype=torch.float32)
        self.slider_armature = torch.tensor(self.slider_armatures, device=device, dtype=torch.float32).expand(num_envs, num_tendons).clone()
        self.j_eff = torch.zeros((num_envs, num_tendons), device=device, dtype=torch.float32)
        self.update_inertia()
        if torch.any(self.j_eff <= 0.0):
            raise RuntimeError("CreateSpool: the slider joints need a positive armature or mass.")
        print(f"CreateSpool: sliders {self.dof_indices_list} (motors {self.motor_indices_list}), armature+mass {self.slider_inertias} kg")

        # Spool state, updated in step (num_envs, num_model_tendons).
        shape = (num_envs, num_tendons)
        self.theta_m = torch.zeros(shape, device=device, dtype=torch.float32)
        self.omega_m = torch.zeros(shape, device=device, dtype=torch.float32)
        self.tension = torch.zeros(shape, device=device, dtype=torch.float32)
        self.tau_e = torch.zeros(shape, device=device, dtype=torch.float32)
        self.just_reset = torch.ones(num_envs, device=device, dtype=torch.bool)
        self.reset_rows = torch.zeros(num_envs, device=device, dtype=torch.bool)  # envs reset before this step
        self.checked_velocity_source = False  # checked on the first step, once update_robot is set up

    def update_inertia(self) -> None:
        """Recompute the rotary inertia J_eff = (armature + slider_mass) * r**2 the simulator integrates."""
        self.j_eff[:] = (self.slider_armature + self.slider_mass) * self.pully_radius**2

    def commanded_angle(self, container: ModuleContainer) -> torch.Tensor:
        """Spool angle targets (rad) of the scaled policy actions, ``(num_envs, num_model_tendons)``."""
        return container.scaled_act_buf[:, container.active_dof_slice][:, self.action_slots]

    def reset(self, container: ModuleContainer) -> None:
        """On the next step use the commanded reset state (and zero tension) for the reset envs."""
        self.just_reset |= container.reset_buf

    def step(self, container: ModuleContainer) -> None:
        """Read the spool state of every model tendon from the simulator."""
        robot = container.robot
        r = self.pully_radius

        if not self.checked_velocity_source:
            self.checked_velocity_source = True
            # The simulator's joint velocity misses the motion the tendon constraint imposes on the
            # slider (it reads ~0 while the slider creeps), so a stop torque built on it cannot hold the spool.
            if robot.velocity_source != "fd":
                print("CreateSpool: WARNING update_robot 'velocity_source' is not 'fd': the stop torque cannot hold the spool.")
        slider_pos = robot.get_joint_pos_buf[:, self.dof_indices]
        slider_vel = robot.joint_velocity[:, self.dof_indices]
        tension = robot.get_tendon_force_buf[:, self.model_tendon_indices].clamp(min=0.0)

        self.reset_rows[:] = self.just_reset
        if bool(torch.any(self.just_reset)):
            just_reset = self.just_reset.unsqueeze(-1)
            slider_pos = torch.where(just_reset, container.reset_joint_pos_buf[:, self.dof_indices], slider_pos)
            slider_vel = torch.where(just_reset, container.reset_joint_vel_buf[:, self.dof_indices], slider_vel)
            tension = torch.where(just_reset, torch.zeros_like(tension), tension)
            self.just_reset[:] = False

        # New tensors every step, so references kept by readers (e.g. last_signals) stay valid.
        self.theta_m = slider_pos / r
        self.omega_m = slider_vel / r
        self.tension = tension
        self.tau_e = -r * tension

"""Servo-motor spool model (M5 friction) driving the tendon actuator sliders (alternative to 'tendon_force_m5')."""

from __future__ import annotations

import torch

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module
from optimal_morphology_rl.modules.robot_control.helpers import build_action_space
from optimal_morphology_rl.modules.tendon_scale_utils import build_tendon_scale_tensor


@register_module("prismatic_force_m5")
class PrismaticForceM5Module(BaseModule):
    """Drives each tendon's prismatic actuator slider as a servo winding the tendon on a spool.

    The vsim generator (urdf_to_vsim.py) prepends a motorized prismatic slider to every actuated
    tendon, with +q pulling the tendon. The slider is the rim of the spool, so the servo state is
    read straight from the simulator instead of being integrated here:

        theta_m = q / r,  omega_m = qdot / r            (winding positive; qdot from 'update_robot')
        tau_e   = -r * T                                 (T: measured tendon tension, previous step)
        tau_m   = motor_gain * (action - theta_m)
        tau_f   = M5 friction, Karnopp stop torque capped by the friction budget
        F       = (tau_m + tau_f) / r                    (motor force applied on the slider DOF)

    The tendon pulls the slider with -T and the servo inertia is the slider's armature, so the
    simulator closes the loop  J_eff * domega_m = tau_m + tau_f + tau_e  with
    J_eff = (armature + slider_mass) * r**2.  The armature should therefore be the reflected rotor
    inertia J / r**2; a mismatch with the 'J' config is reported at startup.

    The module builds the tendon action space itself (replaces 'robot_control_tendons' and
    'apply_tendon_forces'); its forces are applied by 'apply_motor_forces'.
    """

    def finalize(self, container: ModuleContainer) -> None:
        if container.get("robot") is None:
            raise RuntimeError(
                "PrismaticForceM5 requires 'robot' in the shared container. "
                "Ensure the 'create_robot' module is listed before 'prismatic_force_m5'."
            )
        if container.get("env") is None:
            raise RuntimeError("PrismaticForceM5 requires 'env' in the shared container.")

        robot = container.robot
        if not robot.use_tendon:
            raise RuntimeError(
                "PrismaticForceM5 only supports tendon-driven robot configurations (the 'create_robot' config must set 'use_tendon: true')."
            )
        build_action_space(container)

        # Default: every policy-controlled tendon is driven through its slider.
        indices = self.config.get("model_tendon_indices", [int(i) for i in container.active_dof_indices.tolist()])
        if not isinstance(indices, (list, tuple)) or len(indices) == 0:
            raise RuntimeError("PrismaticForceM5 config 'model_tendon_indices' must be a non-empty list of tendon indices.")
        active = [int(i) for i in container.active_dof_indices.tolist()]
        for idx in indices:
            if idx not in active:
                raise RuntimeError(f"PrismaticForceM5 'model_tendon_indices' entry {idx} is not a policy-controlled tendon {active}.")
        self.model_tendon_indices = list(indices)
        self.action_slots_list = [active.index(idx) for idx in self.model_tendon_indices]

        # Map each tendon to its slider motor and DOF by the generator's naming convention.
        suffix = str(self.config.get("slider_joint_suffix", "_slider_joint"))
        art_def = robot.art_def
        motor_by_joint = {art_def.get_motor_def(i).joint_name: i for i in range(robot.num_motors)}
        link_mass = {art_def.get_link_def(i).name: art_def.get_link_def(i).mass for i in range(robot.num_links)}
        self.motor_indices_list = []
        self.dof_indices_list = []
        self.slider_inertias = []  # armature + slider mass, kg
        for idx in self.model_tendon_indices:
            tendon_name = art_def.get_spatial_tendon_def(idx).name
            joint_name = tendon_name + suffix
            if joint_name not in motor_by_joint:
                raise RuntimeError(
                    f"PrismaticForceM5: no motor on slider joint '{joint_name}' for tendon '{tendon_name}'. "
                    "Regenerate the .vsim with ADD_TENDON_SLIDERS = True."
                )
            motor_idx = motor_by_joint[joint_name]
            dof_idx = int(art_def.get_motor_def(motor_idx).dof_index)
            armature = float(art_def.get_joint_dof_def(dof_idx).armature)
            slider_mass = float(link_mass.get(tendon_name + suffix.removesuffix("_joint"), 0.0))
            self.motor_indices_list.append(motor_idx)
            self.dof_indices_list.append(dof_idx)
            self.slider_inertias.append(armature + slider_mass)

    def post_finalize(self, container: ModuleContainer) -> None:
        robot = container.robot
        device = container.device
        num_envs = container.total_num_envs
        num_model_tendons = len(self.model_tendon_indices)

        timestep = container.get("timestep")
        frame_skip = container.get("frame_skip")
        if timestep is None or frame_skip is None:
            raise RuntimeError(
                "PrismaticForceM5 requires 'timestep' and 'frame_skip' in the shared container. "
                "Ensure the 'create_rigid_vsim_envs' module is listed before 'prismatic_force_m5'."
            )
        self.dt = float(timestep) * float(frame_skip)

        def coef(name, default):
            return build_tendon_scale_tensor(self.config.get(name, default), num_model_tendons, device, name)

        self.pully_radius = coef("pully_radius", 0.011)
        self.motor_gain = coef("motor_gain", 0.105)
        self.inertia = coef("J", 0.0011779299150459918)  # reflected: J_rotor * N**2 (reference only)
        self.k_v = coef("K_v", 0.0037641)
        self.k_c = coef("K_c", 0.00666241)
        self.k_m = coef("K_m", 0.447678)
        self.k_e = coef("K_e", 0.00129603)
        self.v_s = coef("v_s", 0.0628076)
        self.alpha = coef("alpha", 1.00722)
        self.k_cs = coef("K_cs", 0.000773996)
        self.k_ms = coef("K_ms", 0.303106)
        self.k_es = coef("K_es", 0.00117)

        if torch.any(self.pully_radius <= 0.0):
            raise RuntimeError("PrismaticForceM5 config 'pully_radius' entries must be positive.")
        if torch.any(self.v_s <= 0.0):
            raise RuntimeError("PrismaticForceM5 config 'v_s' entries must be positive.")

        self.action_slots = torch.tensor(self.action_slots_list, device=device, dtype=torch.long)
        self.motor_indices = torch.tensor(self.motor_indices_list, device=device, dtype=torch.long)
        self.dof_indices = torch.tensor(self.dof_indices_list, device=device, dtype=torch.long)

        # Rotary inertia the simulator integrates for the Karnopp stop torque.
        slider_inertia = torch.tensor(self.slider_inertias, device=device, dtype=torch.float32)
        self.j_eff = slider_inertia * self.pully_radius**2
        if torch.any(self.j_eff <= 0.0):
            raise RuntimeError("PrismaticForceM5: the slider joints need a positive armature or mass.")
        expected_armature = self.inertia / self.pully_radius**2
        print(
            f"PrismaticForceM5: sliders {self.dof_indices_list} (motors {self.motor_indices_list}), "
            f"armature+mass {slider_inertia.cpu().numpy()} kg, J/r^2 from config {expected_armature.cpu().numpy()} kg"
        )

        # Slider velocity source. "fd": the finite difference of the joint positions computed by
        # 'update_robot' (robot.calculated_joint_velocity, default). The simulator's joint velocity
        # misses the motion the tendon constraint imposes on the slider (it reads ~0 while the
        # slider creeps), so a stop torque built on it cannot hold the spool.
        # "sim": the simulator's joint velocity buffer.
        self.velocity_source = str(self.config.get("velocity_source", "fd"))
        if self.velocity_source not in ("fd", "sim"):
            raise RuntimeError(f"PrismaticForceM5 config 'velocity_source' must be 'fd' or 'sim', got '{self.velocity_source}'.")

        container.prismatic_motor_force_buf = torch.zeros((num_envs, robot.num_motors), device=device, dtype=torch.float32)
        self.just_reset = torch.ones(num_envs, device=device, dtype=torch.bool)

        # Optional debug print of one (env, tendon) pair. Off by default: .item() forces a GPU sync.
        self.debug = bool(self.config.get("debug", False))
        self.debug_env = int(self.config.get("debug_env", 0))
        self.debug_tendon = int(self.config.get("debug_tendon", 0))

    def reset(self, container: ModuleContainer) -> None:
        """The read buffers still show the pre-reset state on the next step: use the reset state instead."""
        self.just_reset |= container.reset_buf

    def _friction_budget(self, tau_m, tau_e, w_abs):
        """M5: directional load-dependent friction with Stribeck decay (paper, eq. 9)."""
        coulomb = self.k_c + (self.k_m * tau_m - self.k_e * tau_e).abs()
        static = torch.exp(-((w_abs / self.v_s) ** self.alpha)) * (self.k_cs + (self.k_ms * tau_m - self.k_es * tau_e).abs())
        return self.k_v * w_abs + coulomb + static

    def step(self, container: ModuleContainer) -> None:
        robot = container.robot
        r = self.pully_radius

        slider_pos = robot.get_joint_pos_buf[:, self.dof_indices]
        if self.velocity_source == "fd":
            slider_vel = robot.calculated_joint_velocity[:, self.dof_indices]
        else:
            slider_vel = robot.get_joint_vel_buf[:, self.dof_indices]
        tension = robot.get_tendon_force_buf[:, self.model_tendon_indices].clamp(min=0.0)

        just_reset = self.just_reset.unsqueeze(-1)
        if torch.any(self.just_reset):
            slider_pos = torch.where(just_reset, container.reset_joint_pos_buf[:, self.dof_indices], slider_pos)
            slider_vel = torch.where(just_reset, container.reset_joint_vel_buf[:, self.dof_indices], slider_vel)
            tension = torch.where(just_reset, torch.zeros_like(tension), tension)
            self.just_reset[:] = False

        theta_m = slider_pos / r
        omega_m = slider_vel / r
        tau_e = -r * tension

        actions = container.scaled_act_buf[:, container.active_dof_slice][:, self.action_slots]
        tau_m = self.motor_gain * (actions - theta_m)

        tau_f_max = self._friction_budget(tau_m, tau_e, omega_m.abs())
        tau_f_stop = -(self.j_eff / self.dt * omega_m + tau_m + tau_e)
        tau_f = torch.clamp(tau_f_stop, -tau_f_max, tau_f_max)

        force = (tau_m + tau_f) / r
        container.prismatic_motor_force_buf[:, self.motor_indices] = force

        if self.debug:
            e, t = 0, 0
            vals = {
                "theta_m": theta_m, "omega_m": omega_m, "T": tension, "tau_e": tau_e,
                "tau_m": tau_m, "tau_f": tau_f, "force": force,
            }
            print(" ".join(f"{name} {v[e, t].item(): .4f}" for name, v in vals.items()))

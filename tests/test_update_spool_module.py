"""Unit tests for update_spool, update_spool_target and the hold modes of create_spool's commanded spool angle change."""

from types import SimpleNamespace

import pytest
import torch

from optimal_morphology_rl.envs.modular_environment import ModularEnvironment
from optimal_morphology_rl.modules.create_spool_module import CreateSpoolModule
from optimal_morphology_rl.modules.update_spool_module import UpdateSpoolModule
from optimal_morphology_rl.modules.update_spool_target_module import UpdateSpoolTargetModule

R = 0.005  # m


def _spool() -> CreateSpoolModule:
    """A one-env, one-tendon spool on slider DOF 0 (the buffers create_spool's post_finalize allocates), hold offset."""
    spool = CreateSpoolModule({})
    spool.hold, spool.dt = "offset", 0.005
    spool.dof_indices, spool.model_tendon_indices, spool.action_slots = torch.tensor([0]), [0], torch.tensor([0])
    spool.pully_radius = torch.full((1, 1), R)
    for name in ("theta_m", "omega_m", "tension", "tau_e", "last_slider_pos", "target"):
        setattr(spool, name, torch.zeros(1, 1))
    spool.just_reset = torch.zeros(1, dtype=torch.bool)
    spool.reset_rows = torch.zeros(1, dtype=torch.bool)
    return spool


def _container(spool, action: float):
    robot = SimpleNamespace(get_joint_pos_buf=torch.zeros(1, 1), get_tendon_force_buf=torch.zeros(1, 1))
    container = SimpleNamespace(
        spool=spool, robot=robot, active_dof_slice=slice(0, 1), scaled_act_buf=torch.full((1, 1), action),
        reset_joint_pos_buf=torch.zeros(1, 1), reset_joint_vel_buf=torch.zeros(1, 1),
    )
    container.get = lambda key, default=None: getattr(container, key, default)  # as ModuleContainer.get
    return container


def _substep(container, angle: float) -> float:
    """One physics substep's update_spool with the spool at ``angle`` (rad); the commanded change."""
    container.robot.get_joint_pos_buf[:] = angle * R
    UpdateSpoolModule().step(container)
    return float(container.spool.commanded_delta_angle(container))


def _control_step(container, target_module, angle: float) -> None:
    """The pre_physics part of a control step with the spool at ``angle``: update_spool_target fixes the target."""
    container.robot.get_joint_pos_buf[:] = angle * R
    target_module.step(container)


def test_update_spool_target_fixes_the_desired_angle_once_per_control_step() -> None:
    spool = _spool()
    container = _container(spool, action=0.5)
    module = UpdateSpoolTargetModule()
    module.finalize(container)
    assert spool.hold == "target"
    _control_step(container, module, 1.0)
    assert float(spool.target) == pytest.approx(1.5)  # the angle now + the action
    container.scaled_act_buf[:] = 9.0  # not taken up before the next control step
    assert _substep(container, 1.0) == pytest.approx(0.5)
    assert _substep(container, 1.2) == pytest.approx(0.3)  # the servo moves toward 1.5
    assert _substep(container, 1.5) == pytest.approx(0.0)
    _control_step(container, module, 1.5)
    assert _substep(container, 1.5) == pytest.approx(9.0)  # the next control step's target


def test_hold_offset_commands_the_action_every_substep() -> None:
    container = _container(_spool(), action=0.5)
    for angle in (1.0, 1.2, 1.5):
        assert _substep(container, angle) == pytest.approx(0.5)


def test_hold_target_after_a_reset_commands_the_action_until_the_next_read() -> None:
    spool = _spool()
    container = _container(spool, action=0.5)
    module = UpdateSpoolTargetModule()
    module.finalize(container)
    _control_step(container, module, 1.0)
    _substep(container, 1.0)
    UpdateSpoolModule().reset(SimpleNamespace(spool=spool, reset_buf=torch.ones(1, dtype=torch.bool)))
    container.scaled_act_buf[:] = 0.0  # the reset action
    assert float(spool.commanded_delta_angle(container)) == 0.0  # not the old target less the old angle
    container.reset_joint_pos_buf[:] = 2.0 * R  # the reset state, which update_robot also reads into its buffers
    _control_step(container, module, 2.0)
    assert _substep(container, 2.0) == pytest.approx(0.0)


def _env_with(frame_skip: int, phase: str, module):
    manager = SimpleNamespace(container=SimpleNamespace(frame_skip=frame_skip), _phase_modules={phase: [module]})
    return SimpleNamespace(module_manager=manager)


def test_physics_rate_modules_must_run_every_physics_step() -> None:
    update = UpdateSpoolModule({"_name": "update_spool"})
    with pytest.raises(ValueError, match="'update_spool' runs every physics step"):
        ModularEnvironment._check_physics_rate(_env_with(10, "pre_physics_step_modules", update))
    ModularEnvironment._check_physics_rate(_env_with(10, "pre_gym_step_modules", update))
    ModularEnvironment._check_physics_rate(_env_with(1, "pre_physics_step_modules", update))  # one substep: either
    ModularEnvironment._check_physics_rate(_env_with(10, "pre_physics_step_modules", UpdateSpoolTargetModule()))

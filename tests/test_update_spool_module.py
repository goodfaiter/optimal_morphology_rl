"""Unit tests for update_spool and the hold modes of create_spool's commanded spool angle change."""

from types import SimpleNamespace

import pytest
import torch

from optimal_morphology_rl.modules.create_spool_module import CreateSpoolModule
from optimal_morphology_rl.modules.update_spool_module import UpdateSpoolModule

R = 0.005  # m


def _spool(hold: str) -> CreateSpoolModule:
    """A one-env, one-tendon spool on slider DOF 0 (the buffers create_spool's post_finalize allocates)."""
    spool = CreateSpoolModule({"hold": hold})
    spool.hold, spool.dt = hold, 0.005
    spool.dof_indices, spool.model_tendon_indices, spool.action_slots = torch.tensor([0]), [0], torch.tensor([0])
    spool.pully_radius = torch.full((1, 1), R)
    for name in ("theta_m", "omega_m", "tension", "tau_e", "last_slider_pos", "target"):
        setattr(spool, name, torch.zeros(1, 1))
    spool.just_reset = torch.zeros(1, dtype=torch.bool)
    spool.reset_rows = torch.zeros(1, dtype=torch.bool)
    return spool


def _container(spool, action: float, frame_skip: int = 10):
    robot = SimpleNamespace(get_joint_pos_buf=torch.zeros(1, 1), get_tendon_force_buf=torch.zeros(1, 1))
    container = SimpleNamespace(
        spool=spool, robot=robot, frame_skip=frame_skip, substep=0, active_dof_slice=slice(0, 1),
        scaled_act_buf=torch.full((1, 1), action), reset_joint_pos_buf=torch.zeros(1, 1), reset_joint_vel_buf=torch.zeros(1, 1),
    )
    container.get = lambda key, default=None: getattr(container, key, default)  # as ModuleContainer.get
    return container


def _substep(module, container, k: int, angle: float) -> torch.Tensor:
    """Run update_spool on substep k with the spool at ``angle`` (rad); return the commanded change."""
    container.substep = k
    container.robot.get_joint_pos_buf[:] = angle * R
    module.step(container)
    return container.spool.commanded_delta_angle(container)


def test_hold_target_fixes_the_desired_angle_on_the_first_substep() -> None:
    spool = _spool("target")
    container = _container(spool, action=0.5)
    module = UpdateSpoolModule()
    assert float(_substep(module, container, 0, 1.0)) == pytest.approx(0.5)  # target = 1.0 + 0.5
    assert float(spool.target) == pytest.approx(1.5)
    container.scaled_act_buf[:] = 9.0  # a changed action is not taken up before the next control step
    assert float(_substep(module, container, 1, 1.2)) == pytest.approx(0.3)  # the servo moves toward 1.5
    assert float(_substep(module, container, 9, 1.5)) == pytest.approx(0.0)
    assert float(_substep(module, container, 0, 1.5)) == pytest.approx(9.0)  # next control step: new target


def test_hold_offset_commands_the_action_every_substep() -> None:
    spool = _spool("offset")
    container = _container(spool, action=0.5)
    module = UpdateSpoolModule()
    for k, angle in ((0, 1.0), (1, 1.2), (9, 1.5)):
        assert float(_substep(module, container, k, angle)) == pytest.approx(0.5)


def test_hold_target_after_a_reset_commands_the_action_until_the_next_read() -> None:
    spool = _spool("target")
    container = _container(spool, action=0.5)
    module = UpdateSpoolModule()
    _substep(module, container, 0, 1.0)
    module.reset(SimpleNamespace(spool=spool, reset_buf=torch.ones(1, dtype=torch.bool)))
    container.scaled_act_buf[:] = 0.0  # the reset action
    assert float(spool.commanded_delta_angle(container)) == 0.0  # not the old target less the old angle
    assert float(_substep(module, container, 0, 2.0)) == pytest.approx(0.0)  # read: the reset state + action 0


def test_update_spool_outside_the_substeps_needs_frame_skip_one() -> None:
    spool = _spool("target")
    container = _container(spool, action=0.5, frame_skip=10)
    container.substep = None
    with pytest.raises(RuntimeError, match="pre_gym_step_modules"):
        UpdateSpoolModule().step(container)
    container.frame_skip = 1  # one substep: pre_physics works too, every step is a new action
    UpdateSpoolModule().step(container)
    assert float(spool.target) == pytest.approx(0.5)

"""Unit tests for the calculated tendon velocity/acceleration in the update_robot module."""

import pytest
import torch

from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.update_robot_module import UpdateRobotModule


class _FakeEnv:
    pass


class _FakeRobot:
    def __init__(self, num_tendons: int = 3):
        self.use_tendon = True
        self.num_tendons = num_tendons
        self.get_tendon_lengths_buf = torch.zeros((1, num_tendons), dtype=torch.float32)
        self.get_tendon_vel_buf = torch.zeros((1, num_tendons), dtype=torch.float32)
        self.get_tendon_force_buf = torch.zeros((1, num_tendons), dtype=torch.float32)
        self.calculated_tendon_velocity = torch.zeros((1, num_tendons), dtype=torch.float32)
        self.calculated_tendon_length_rate = torch.zeros((1, num_tendons), dtype=torch.float32)
        self.calculated_tendon_acceleration = torch.zeros((1, num_tendons), dtype=torch.float32)


@pytest.fixture
def container() -> ModuleContainer:
    cont = ModuleContainer()
    cont.total_num_envs = 1
    cont.device = torch.device("cpu")
    cont.env = _FakeEnv()
    cont.robot = _FakeRobot()
    cont.timestep = 0.005  # control dt = timestep * frame_skip = 0.01
    cont.frame_skip = 2
    cont.reset_buf = torch.zeros(1, dtype=torch.bool)
    cont.robot.get_tendon_lengths_buf = torch.tensor([[2.0, 3.0, 9.0]])
    return cont


def _make_module(**overrides) -> UpdateRobotModule:
    config = {
        "max_tendon_speed": 20.0,  # joint speed limit; phantom jumps exceed it
    }
    config.update(overrides)
    return UpdateRobotModule(config)


def _setup(container: ModuleContainer, module: UpdateRobotModule) -> None:
    module._setup_tendon_dynamics(container)


def _compute(container: ModuleContainer, module: UpdateRobotModule) -> None:
    module._compute_tendon_dynamics(container.robot)


def test_setup_allocates_filter_state(container: ModuleContainer) -> None:
    module = _make_module()
    _setup(container, module)

    assert module.dt == pytest.approx(0.01)
    assert module.prev_length.shape == (1, 3)
    assert module.prev_vel.shape == (1, 3)
    assert module.prev_acc.shape == (1, 3)
    # No length history yet: the first step uses zero velocity.
    assert torch.all(torch.isnan(module.prev_length))
    assert torch.all(module.prev_vel == 0.0)
    assert torch.all(module.prev_acc == 0.0)
    # Default filter alphas are 1.0 (raw).
    assert torch.all(module.acc_filter_alpha == 1.0)
    assert torch.all(module.vel_filter_alpha == 1.0)


def test_setup_supports_per_tendon_filters(container: ModuleContainer) -> None:
    module = _make_module(vel_filter_alpha=[1.0, 0.5, 0.0])
    _setup(container, module)

    assert torch.all(module.vel_filter_alpha == torch.tensor([1.0, 0.5, 0.0]))
    assert module.vel_filter_alpha.shape == (3,)


def test_step_computes_velocity_via_finite_difference(container: ModuleContainer) -> None:
    # Tendon velocity is the negated length finite difference (winding convention);
    # the simulator velocity buffer is ignored.
    module = _make_module()
    _setup(container, module)

    _compute(container, module)  # no length history yet -> vel = 0 -> acc = 0
    assert torch.all(container.robot.calculated_tendon_velocity == 0.0)
    assert torch.all(container.robot.calculated_tendon_acceleration == 0.0)

    # Lengths move by -0.01 (tendon 0) and +0.02 (tendon 1) over dt = 0.01.
    container.robot.get_tendon_lengths_buf = torch.tensor([[1.99, 3.02, 9.0]])
    _compute(container, module)

    # Tendon 0: vel = +1 -> acc = 100. Tendon 1: vel = -2 -> acc = -200.
    assert container.robot.calculated_tendon_velocity[0, 0].item() == pytest.approx(1.0)
    assert container.robot.calculated_tendon_velocity[0, 1].item() == pytest.approx(-2.0)
    assert container.robot.calculated_tendon_velocity[0, 2].item() == 0.0
    assert container.robot.calculated_tendon_acceleration[0, 0].item() == pytest.approx(100.0)
    assert container.robot.calculated_tendon_acceleration[0, 1].item() == pytest.approx(-200.0)


def test_step_filters_velocity(container: ModuleContainer) -> None:
    # alpha = 0.5: vel = 0.5*vel_raw + 0.5*prev_filtered (prev starts at 0),
    # raw acceleration (acc_filter_alpha = 1.0).
    module = _make_module(vel_filter_alpha=0.5)
    _setup(container, module)

    _compute(container, module)  # first step: no history -> vel = 0
    assert torch.all(container.robot.calculated_tendon_velocity == 0.0)

    # Length grows by 0.01 (unwinding): vel_raw = -1 -> filtered -0.5.
    container.robot.get_tendon_lengths_buf = torch.tensor([[2.01, 3.0, 9.0]])
    _compute(container, module)
    assert container.robot.calculated_tendon_velocity[0, 0].item() == pytest.approx(-0.5)
    assert container.robot.calculated_tendon_acceleration[0, 0].item() == pytest.approx(-50.0)

    # Same length: vel_raw = 0 -> filtered = 0.5*0 + 0.5*(-0.5) = -0.25 -> acc_raw = 25.
    _compute(container, module)
    assert container.robot.calculated_tendon_velocity[0, 0].item() == pytest.approx(-0.25)
    assert container.robot.calculated_tendon_acceleration[0, 0].item() == pytest.approx(25.0)


def test_step_filters_acceleration(container: ModuleContainer) -> None:
    # alpha = 0.5: acc = 0.5*acc_raw + 0.5*prev_filtered (prev starts at 0).
    module = _make_module(acc_filter_alpha=0.5)
    _setup(container, module)

    _compute(container, module)  # first step: no history -> vel = 0 -> acc = 0
    assert torch.all(container.robot.calculated_tendon_acceleration == 0.0)

    # Length grows by 0.01: vel = -1 -> acc_raw = -100 -> filtered -50.
    container.robot.get_tendon_lengths_buf = torch.tensor([[2.01, 3.0, 9.0]])
    _compute(container, module)
    assert container.robot.calculated_tendon_acceleration[0, 0].item() == pytest.approx(-50.0)

    # Constant velocity: acc_raw = 0 -> filtered = 0.5*0 + 0.5*(-50) = -25.
    container.robot.get_tendon_lengths_buf = torch.tensor([[2.02, 3.0, 9.0]])
    _compute(container, module)
    assert container.robot.calculated_tendon_acceleration[0, 0].item() == pytest.approx(-25.0)


def test_step_soft_deadband_zeroes_noise_floor_velocity(container: ModuleContainer) -> None:
    # vel_deadband attenuates the filtered velocity by eps (soft, continuous):
    # velocity jitter inside the noise floor fully zeroes, so the acceleration
    # finite difference cannot amplify it into chatter at near-zero velocity.
    module = _make_module(vel_deadband=0.5)
    _setup(container, module)

    _compute(container, module)  # first step: no history -> vel = 0
    assert torch.all(container.robot.calculated_tendon_velocity == 0.0)

    # Length grows by 0.003: vel_raw = -0.3 -> |vel| < 0.5 -> fully zeroed.
    container.robot.get_tendon_lengths_buf = torch.tensor([[2.003, 3.0, 9.0]])
    _compute(container, module)
    assert torch.all(module.prev_vel == 0.0)
    assert container.robot.calculated_tendon_acceleration[0, 0].item() == 0.0

    # Length grows by 0.01: vel_raw = -0.7 -> soft-debanded to -(0.7 - 0.5) = -0.2.
    container.robot.get_tendon_lengths_buf = torch.tensor([[2.01, 3.0, 9.0]])
    _compute(container, module)
    assert module.prev_vel[0, 0].item() == pytest.approx(-0.2, abs=1e-4)
    assert container.robot.calculated_tendon_acceleration[0, 0].item() == pytest.approx(-20.0, rel=1e-4)


def test_step_soft_deadband_on_acc(container: ModuleContainer) -> None:
    # acc_deadband attenuates the filtered acceleration by eps (soft, continuous):
    # small alternating accelerations zero out.
    module = _make_module(acc_deadband=25.0)
    _setup(container, module)

    _compute(container, module)  # first step: vel = 0 -> acc_raw = 0 -> acc = 0
    assert torch.all(module.prev_acc == 0.0)

    # Length grows by 0.01: vel = -1 -> acc_raw = -100 -> soft-debanded to -75.
    container.robot.get_tendon_lengths_buf = torch.tensor([[2.01, 3.0, 9.0]])
    _compute(container, module)
    assert module.prev_acc[0, 0].item() == pytest.approx(-75.0, rel=1e-5)
    assert container.robot.calculated_tendon_acceleration[0, 0].item() == pytest.approx(-75.0, rel=1e-5)

    # Constant velocity (length keeps growing by 0.01): acc_raw = 0 -> acc stays 0.
    container.robot.get_tendon_lengths_buf = torch.tensor([[2.02, 3.0, 9.0]])
    _compute(container, module)
    assert torch.all(module.prev_acc == 0.0)


def test_step_spike_guard_rejects_teleport_jumps(container: ModuleContainer) -> None:
    # max_tendon_speed rejects phantom velocities from the reset teleport: a length
    # jump far beyond the physical speed limit zeroes the raw velocity.
    module = _make_module(max_tendon_speed=1.5)
    _setup(container, module)

    _compute(container, module)  # first step: vel = 0
    assert container.robot.calculated_tendon_velocity[0, 0].item() == 0.0

    # The reset teleport: the length jumps by 0.02 m in one step (vel_raw = -2 >> 1.5).
    container.robot.get_tendon_lengths_buf = torch.tensor([[2.02, 3.0, 9.0]])
    _compute(container, module)
    assert torch.all(module.prev_vel == 0.0)
    assert container.robot.calculated_tendon_velocity[0, 0].item() == 0.0

    # The next step resumes the normal FD from the fresh length.
    container.robot.get_tendon_lengths_buf = torch.tensor([[2.03, 3.0, 9.0]])
    _compute(container, module)
    assert module.prev_vel[0, 0].item() == pytest.approx(-1.0, abs=1e-4)
    assert container.robot.calculated_tendon_acceleration[0, 0].item() == pytest.approx(-100.0)


def test_reset_clears_velocity_history(container: ModuleContainer) -> None:
    module = _make_module()
    _setup(container, module)

    _compute(container, module)  # first step: vel = 0 -> acc = 0

    container.reset_buf[0] = True
    module.reset(container)
    assert torch.all(torch.isnan(module.prev_length))

    # First step after reset uses vel = 0 (NaN length history) -> acc = 0
    # (with uncleared history it would be vel = -1 -> acc = -100).
    container.robot.get_tendon_lengths_buf = torch.tensor([[2.01, 3.0, 9.0]])
    _compute(container, module)
    assert container.robot.calculated_tendon_velocity[0, 0].item() == 0.0
    assert container.robot.calculated_tendon_acceleration[0, 0].item() == 0.0


def test_reset_clears_filtered_acc(container: ModuleContainer) -> None:
    module = _make_module(acc_filter_alpha=0.5)
    _setup(container, module)

    container.robot.get_tendon_lengths_buf = torch.tensor([[1.99, 3.0, 9.0]])
    _compute(container, module)  # first step: vel = +1 -> acc = 0

    # Second step: vel = -1 -> acc_raw = -200 -> filtered = -50.
    container.robot.get_tendon_lengths_buf = torch.tensor([[2.00, 3.0, 9.0]])
    _compute(container, module)
    assert container.robot.calculated_tendon_acceleration[0, 0].item() == pytest.approx(-50.0)

    container.reset_buf[0] = True
    module.reset(container)
    _compute(container, module)

    # First step after reset uses vel = 0 and the filter state is cleared:
    # filtered = 0.5*0 + 0.5*0 = 0 (not 0.5*0 + 0.5*(-50) = -25).
    assert container.robot.calculated_tendon_acceleration[0, 0].item() == 0.0


def test_reset_is_selective_per_env(container: ModuleContainer) -> None:
    container.total_num_envs = 2
    container.robot.get_tendon_lengths_buf = torch.tensor([[2.0, 3.0, 9.0], [2.0, 3.0, 9.0]])
    container.robot.calculated_tendon_velocity = torch.zeros((2, 3))
    container.robot.calculated_tendon_length_rate = torch.zeros((2, 3))
    container.robot.calculated_tendon_acceleration = torch.zeros((2, 3))
    module = _make_module()
    _setup(container, module)

    _compute(container, module)
    # Both envs build velocity/acceleration history.
    container.robot.get_tendon_lengths_buf = torch.tensor([[2.01, 3.02, 9.0], [2.01, 3.02, 9.0]])
    _compute(container, module)

    container.reset_buf = torch.tensor([True, False])
    module.reset(container)

    # Env 0: all history cleared.
    assert torch.all(torch.isnan(module.prev_length[0]))
    assert torch.all(module.prev_vel[0] == 0.0)
    assert torch.all(module.prev_acc[0] == 0.0)
    # Env 1: history preserved.
    assert torch.all(torch.isfinite(module.prev_length[1]))
    assert torch.allclose(module.prev_length[1], torch.tensor([2.01, 3.02, 9.0]))
    assert module.prev_vel[1, 0].item() != 0.0
    assert module.prev_acc[1, 0].item() != 0.0

    # Env 0's next step uses zero velocity (NaN guard keeps the first-step velocity at 0);
    # env 1 continues from its history.
    container.robot.get_tendon_lengths_buf = torch.tensor([[2.02, 3.04, 9.0], [2.02, 3.04, 9.0]])
    _compute(container, module)
    assert container.robot.calculated_tendon_velocity[0, 0].item() == 0.0
    assert container.robot.calculated_tendon_velocity[1, 0].item() == pytest.approx(-1.0, abs=1e-4)


def test_setup_requires_timestep_and_frame_skip(container: ModuleContainer) -> None:
    del container._data["timestep"]
    module = _make_module()
    with pytest.raises(RuntimeError, match="'timestep' and 'frame_skip'"):
        module._setup_tendon_dynamics(container)


def test_setup_rejects_nonpositive_max_tendon_speed(container: ModuleContainer) -> None:
    module = _make_module(max_tendon_speed=0.0)
    with pytest.raises(RuntimeError, match="max_tendon_speed"):
        module._setup_tendon_dynamics(container)


def test_setup_rejects_out_of_range_vel_filter_alpha(container: ModuleContainer) -> None:
    module = _make_module(vel_filter_alpha=1.5)
    with pytest.raises(RuntimeError, match="vel_filter_alpha"):
        module._setup_tendon_dynamics(container)


def test_setup_rejects_out_of_range_acc_filter_alpha(container: ModuleContainer) -> None:
    module = _make_module(acc_filter_alpha=1.5)
    with pytest.raises(RuntimeError, match="acc_filter_alpha"):
        module._setup_tendon_dynamics(container)


def test_setup_rejects_negative_vel_deadband(container: ModuleContainer) -> None:
    module = _make_module(vel_deadband=-0.1)
    with pytest.raises(RuntimeError, match="vel_deadband"):
        module._setup_tendon_dynamics(container)


def test_setup_rejects_negative_acc_deadband(container: ModuleContainer) -> None:
    module = _make_module(acc_deadband=-1.0)
    with pytest.raises(RuntimeError, match="acc_deadband"):
        module._setup_tendon_dynamics(container)


def test_length_rate_is_negated_tendon_velocity(container: ModuleContainer) -> None:
    module = _make_module()
    _setup(container, module)
    _compute(container, module)
    container.robot.get_tendon_lengths_buf = torch.tensor([[2.01, 2.98, 9.0]])
    _compute(container, module)

    # dL/dt of the lengths, the opposite sign of the winding-positive calculated_tendon_velocity.
    expected = torch.tensor([[1.0, -2.0, 0.0]])
    assert torch.allclose(container.robot.calculated_tendon_length_rate, expected, atol=1e-4)
    assert torch.allclose(container.robot.calculated_tendon_length_rate, -container.robot.calculated_tendon_velocity)

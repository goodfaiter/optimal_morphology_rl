"""Unit tests for the tendon_force_m5 virtual-motor tendon module."""

import pytest
import torch

from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.tendon_force_m5_module import TendonForceM5Module


class _FakeEnv:
    pass


class _FakeRobot:
    def __init__(self, num_tendons: int = 3):
        self.use_tendon = True
        self.num_tendons = num_tendons
        self.get_tendon_lengths_buf = torch.zeros((1, num_tendons), dtype=torch.float32)
        self.get_tendon_vel_buf = torch.zeros((1, num_tendons), dtype=torch.float32)
        self.get_tendon_force_buf = torch.zeros((1, num_tendons), dtype=torch.float32)


@pytest.fixture
def container() -> ModuleContainer:
    cont = ModuleContainer()
    cont.total_num_envs = 1
    cont.device = torch.device("cpu")
    cont.env = _FakeEnv()
    cont.robot = _FakeRobot()
    cont.timestep = 0.005  # control dt = timestep * frame_skip = 0.01
    cont.frame_skip = 2
    cont.active_dof_slice = slice(0, 3)
    cont.active_dof_indices = torch.tensor([0, 1, 2])
    cont.scaled_act_buf = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    cont.reset_buf = torch.zeros(1, dtype=torch.bool)
    cont.robot.get_tendon_lengths_buf = torch.tensor([[2.0, 3.0, 9.0]])
    cont.robot.get_tendon_force_buf = torch.tensor([[0.5, 0.2, 0.9]])
    return cont


def _make_module(**overrides) -> TendonForceM5Module:
    """Module with simple coefficients and radius 1.0 so theta = -length, omega = vel."""
    config = {
        "model_tendon_indices": [0, 1],
        "pully_radius": 1.0,
        "max_tendon_speed": 20.0,  # 20 rad/s joint limit at the test radius
        "motor_gain": 2.0,
        "J": 1.0,
        "K_v": 0.5,
        "K_c": 0.1,
        "K_m": 0.7,
        "K_e": 0.3,
        "v_s": 0.5,
        "alpha": 2.0,
        "K_cs": 0.05,
        "K_ms": 0.6,
        "K_es": 0.2,
    }
    config.update(overrides)
    return TendonForceM5Module(config)


def _finalize(container: ModuleContainer, module: TendonForceM5Module) -> None:
    module.finalize(container)
    module.post_finalize(container)


def test_post_finalize_allocates_buffers(container: ModuleContainer) -> None:
    module = _make_module()
    _finalize(container, module)

    assert container.tendon_force_buf.shape == (1, 3)
    assert torch.equal(container.tendon_model_indices, torch.tensor([0, 1]))
    assert module.dt == pytest.approx(0.01)
    assert module.prev_length.shape == (1, 2)
    assert module.prev_vel.shape == (1, 2)
    assert module.prev_acc.shape == (1, 2)
    # No length history yet: the first step uses zero velocity.
    assert torch.all(torch.isnan(module.prev_length))
    assert torch.all(module.prev_vel == 0.0)
    assert torch.all(module.prev_acc == 0.0)
    # Default filter alphas are 1.0 (raw).
    assert torch.all(module.acc_filter_alpha == 1.0)
    assert torch.all(module.vel_filter_alpha == 1.0)


def test_step_stick_regime_keeps_previous_force(container: ModuleContainer) -> None:
    # First step (no length history): vel = 0, acc = 0, and |tau_m + tau_e_prev| <= tau_f_max:
    # friction absorbs the stop-torque exactly, so tau_e = tau_e_prev (force = sim force).
    module = _make_module()
    _finalize(container, module)
    module.step(container)

    # Tendon 0: theta = -2, tau_m = 2*(1+2) = 6, tau_e_prev = -0.5,
    # tau_f_max = 0.1 + |4.2 + 0.15| + (0.05 + |3.6 + 0.1|) = 8.2, stop = -5.5 (stick).
    # Tendon 1: theta = -3, tau_m = 10, tau_e_prev = -0.2, tau_f_max = 13.25, stop = -9.8 (stick).
    assert container.tendon_force_buf[:, 0].item() == pytest.approx(0.5, rel=1e-5)
    assert container.tendon_force_buf[:, 1].item() == pytest.approx(0.2, rel=1e-5)
    # The third tendon is never written by the module.
    assert container.tendon_force_buf[:, 2].item() == 0.0


def test_step_computes_velocity_via_finite_difference(container: ModuleContainer) -> None:
    # Tendon velocity is the negated length finite difference (winding convention);
    # the simulator velocity buffer is ignored.
    # Huge friction cap (K_c) so tau_f is unclamped: tau_e = J*acc + J/dt*omega + tau_e_prev.
    module = _make_module(K_m=0.0, K_e=0.0, K_ms=0.0, K_es=0.0, K_c=1000.0)
    _finalize(container, module)

    container.robot.get_tendon_vel_buf = torch.full((1, 3), 999.0)  # ignored by the module
    module.step(container)  # no length history yet -> vel = 0 -> acc = 0 -> force = sim force
    assert container.tendon_force_buf[:, 0].item() == pytest.approx(0.5, rel=1e-5)
    assert container.tendon_force_buf[:, 1].item() == pytest.approx(0.2, rel=1e-5)

    # Lengths move by -0.01 (tendon 0) and +0.02 (tendon 1) over dt = 0.01.
    container.robot.get_tendon_lengths_buf = torch.tensor([[1.99, 3.02, 9.0]])
    module.step(container)

    # Tendon 0: vel = +1 -> omega = 1, omega_prev = 0 -> acc = 100
    # -> tau_e = 100 + 100 - 0.5 = 199.5 -> force = -199.5 -> clamped to 0.
    assert container.tendon_force_buf[:, 0].item() == 0.0
    # Tendon 1: vel = -2 -> omega = -2 -> acc = -200
    # -> tau_e = -200 - 200 - 0.2 = -400.2 -> force = 400.2.
    assert container.tendon_force_buf[:, 1].item() == pytest.approx(400.2, rel=1e-5)


def test_step_positive_force_via_acceleration(container: ModuleContainer) -> None:
    # tau_e = J*acc - tau_m - tau_f with acc from the finite-difference velocity;
    # applied force is -tau_e/radius (vsim convention), so negative tau_e pulls.
    module = _make_module(K_m=0.0, K_e=0.0, K_ms=0.0, K_es=0.0, K_c=1000.0)
    _finalize(container, module)

    container.robot.get_tendon_force_buf = torch.zeros((1, 3))
    container.scaled_act_buf = torch.tensor([[-2.0, 2.0, 3.0]], dtype=torch.float32)
    container.robot.get_tendon_lengths_buf = torch.tensor([[2.0, 3.0, 9.0]])
    module.step(container)  # first step: no history -> vel = 0 -> acc = 0 -> force = 0
    assert container.tendon_force_buf[:, 0].item() == 0.0

    # Length grows by 0.01 (unwinding): vel = -1 -> omega = -1, omega_prev = 0 -> acc = -100
    # -> tau_e = -100 - 100 = -200 -> force = 200.
    container.robot.get_tendon_lengths_buf = torch.tensor([[2.01, 3.0, 9.0]])
    module.step(container)
    assert container.tendon_force_buf[:, 0].item() == pytest.approx(200.0, rel=1e-5)

    # Same length: vel = 0, omega_prev = -1 -> acc = +100 -> tau_e = 100 -> force clamped to 0.
    module.step(container)
    assert container.tendon_force_buf[:, 0].item() == 0.0


def test_step_uses_updated_velocity_history(container: ModuleContainer) -> None:
    module = _make_module(K_m=0.0, K_e=0.0, K_ms=0.0, K_es=0.0, K_c=1000.0)
    _finalize(container, module)

    container.robot.get_tendon_force_buf = torch.zeros((1, 3))
    container.scaled_act_buf = torch.tensor([[-2.0, 2.0, 3.0]], dtype=torch.float32)

    container.robot.get_tendon_lengths_buf = torch.tensor([[2.0, 3.0, 9.0]])
    module.step(container)  # first step: vel = 0 -> force = 0
    assert container.tendon_force_buf[:, 0].item() == 0.0

    # Step 2: length history from step 1 -> vel = -1 -> omega = -1, omega_prev = 0 -> acc = -100.
    container.robot.get_tendon_lengths_buf = torch.tensor([[2.01, 3.0, 9.0]])
    module.step(container)
    assert container.tendon_force_buf[:, 0].item() == pytest.approx(200.0, rel=1e-5)

    # Step 3: vel = 0, omega_prev = -1 -> acc = +100 -> tau_e = 100 -> force clamped to 0.
    module.step(container)
    assert container.tendon_force_buf[:, 0].item() == 0.0


def test_step_filters_velocity(container: ModuleContainer) -> None:
    # alpha = 0.5: vel = 0.5*vel_raw + 0.5*prev_filtered (prev starts at 0),
    # raw acceleration (acc_filter_alpha = 1.0). Huge friction cap so
    # tau_f is unclamped: force = clamp(-(J*acc + J/dt*omega + tau_e_prev), min=0).
    module = _make_module(K_m=0.0, K_e=0.0, K_ms=0.0, K_es=0.0, K_c=1000.0, vel_filter_alpha=0.5)
    _finalize(container, module)

    container.robot.get_tendon_force_buf = torch.zeros((1, 3))
    container.scaled_act_buf = torch.tensor([[-2.0, 2.0, 3.0]], dtype=torch.float32)
    module.step(container)  # first step: no history -> vel = 0 -> acc = 0 -> force = 0
    assert container.tendon_force_buf[:, 0].item() == 0.0

    # Length grows by 0.01 (unwinding): vel_raw = -1 -> filtered -0.5 -> omega = -0.5,
    # omega_prev = 0 -> acc_raw = -50 -> tau_e = -50 - 50 = -100 -> force = 100
    # (with the raw velocity it would be -100 - 100 = -200 -> force = 200).
    container.robot.get_tendon_lengths_buf = torch.tensor([[2.01, 3.0, 9.0]])
    module.step(container)
    assert container.tendon_force_buf[:, 0].item() == pytest.approx(100.0, rel=1e-5)

    # Same length: vel_raw = 0 -> filtered = 0.5*0 + 0.5*(-0.5) = -0.25, omega_prev = -0.5
    # -> acc_raw = 25 -> tau_e = 25 - 25 = 0 -> force = 0.
    module.step(container)
    assert container.tendon_force_buf[:, 0].item() == 0.0


def test_reset_clears_velocity_history(container: ModuleContainer) -> None:
    module = _make_module()
    _finalize(container, module)

    container.robot.get_tendon_force_buf = torch.zeros((1, 3))
    container.scaled_act_buf = torch.tensor([[-2.0, 2.0, 3.0]], dtype=torch.float32)
    container.robot.get_tendon_lengths_buf = torch.tensor([[2.0, 3.0, 9.0]])
    module.step(container)  # first step: vel = 0 -> acc = 0 -> force = 0

    container.reset_buf[0] = True
    module.reset(container)
    assert torch.all(torch.isnan(module.prev_length))

    # First step after reset uses vel = 0 (NaN length history) -> acc = 0 -> force = 0
    # (with uncleared history it would be vel = -1 -> acc = -100 -> force ~ 200).
    container.robot.get_tendon_lengths_buf = torch.tensor([[2.01, 3.0, 9.0]])
    module.step(container)

    assert container.tendon_force_buf[:, 0].item() == 0.0


def test_step_supports_multiple_envs(container: ModuleContainer) -> None:
    container.total_num_envs = 2
    module = _make_module()
    _finalize(container, module)

    container.scaled_act_buf = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=torch.float32)
    container.robot.get_tendon_lengths_buf = torch.tensor([[2.0, 3.0, 9.0], [2.0, 3.0, 9.0]])
    container.robot.get_tendon_force_buf = torch.tensor([[0.5, 0.2, 0.9], [0.5, 0.2, 0.9]])
    module.step(container)

    # Identical inputs produce identical rows for both envs (stick regime keeps the sim force).
    assert torch.allclose(container.tendon_force_buf[:, 0], torch.full((2,), 0.5))
    assert torch.allclose(container.tendon_force_buf[:, 1], torch.full((2,), 0.2))
    assert torch.all(container.tendon_force_buf[:, 2] == 0.0)


def test_step_supports_per_tendon_coefficients(container: ModuleContainer) -> None:
    # Per-tendon motor gains with a constant friction cap (K_c + K_cs = 0.15):
    # the saturated friction leaves tau_e = -tau_m + 0.15, so the gain shows in the force.
    module = _make_module(K_m=0.0, K_e=0.0, K_ms=0.0, K_es=0.0, motor_gain=[2.0, 4.0])
    _finalize(container, module)

    container.scaled_act_buf = torch.tensor([[3.0, 3.0, 3.0]], dtype=torch.float32)
    module.step(container)

    # Tendon 0: tau_m = 2*3 = 6 -> tau_f = -0.15 -> tau_e = -5.85 -> force = 5.85.
    # Tendon 1: tau_m = 4*3 = 12 -> tau_f = -0.15 -> tau_e = -11.85 -> force = 11.85.
    assert container.tendon_force_buf[:, 0].item() == pytest.approx(5.85, rel=1e-5)
    assert container.tendon_force_buf[:, 1].item() == pytest.approx(11.85, rel=1e-5)


def test_step_filters_acceleration(container: ModuleContainer) -> None:
    # alpha = 0.5: acc = 0.5*acc_raw + 0.5*prev_filtered (prev starts at 0).
    module = _make_module(K_m=0.0, K_e=0.0, K_ms=0.0, K_es=0.0, K_c=1000.0, acc_filter_alpha=0.5)
    _finalize(container, module)

    container.robot.get_tendon_force_buf = torch.zeros((1, 3))
    container.scaled_act_buf = torch.tensor([[-2.0, 2.0, 3.0]], dtype=torch.float32)
    module.step(container)  # first step: no history -> vel = 0 -> acc = 0 -> force = 0
    assert container.tendon_force_buf[:, 0].item() == 0.0

    # Length grows by 0.01 (unwinding): vel = -1 -> omega = -1, omega_prev = 0
    # -> acc_raw = -100 -> filtered -50 -> tau_e = -50 - 100 = -150 -> force = 150.
    container.robot.get_tendon_lengths_buf = torch.tensor([[2.01, 3.0, 9.0]])
    module.step(container)
    assert container.tendon_force_buf[:, 0].item() == pytest.approx(150.0, rel=1e-5)

    # Constant velocity: acc_raw = 0 -> filtered = 0.5*0 + 0.5*(-50) = -25
    # -> tau_e = -25 - 100 = -125 -> force = 125.
    container.robot.get_tendon_lengths_buf = torch.tensor([[2.02, 3.0, 9.0]])
    module.step(container)
    assert container.tendon_force_buf[:, 0].item() == pytest.approx(125.0, rel=1e-5)


def test_reset_clears_filtered_acc(container: ModuleContainer) -> None:
    module = _make_module(K_m=0.0, K_e=0.0, K_ms=0.0, K_es=0.0, K_c=1000.0, acc_filter_alpha=0.5)
    _finalize(container, module)

    container.robot.get_tendon_force_buf = torch.zeros((1, 3))
    container.scaled_act_buf = torch.tensor([[-2.0, 2.0, 3.0]], dtype=torch.float32)
    container.robot.get_tendon_lengths_buf = torch.tensor([[1.99, 3.0, 9.0]])
    module.step(container)  # first step: vel = 0 -> acc = 0 -> force = 0

    # Second step: vel = -1 -> acc_raw = -100 -> filtered = -50 -> force = 150.
    container.robot.get_tendon_lengths_buf = torch.tensor([[2.00, 3.0, 9.0]])
    module.step(container)
    assert container.tendon_force_buf[:, 0].item() == pytest.approx(150.0, rel=1e-5)

    container.reset_buf[0] = True
    module.reset(container)
    module.step(container)

    # First step after reset uses vel = 0 and the filter state is cleared:
    # filtered = 0.5*0 + 0.5*0 = 0 -> force = 0 (not 0.5*0 + 0.5*(-50) = -25 -> 125).
    assert container.tendon_force_buf[:, 0].item() == 0.0


def test_step_soft_deadband_zeroes_noise_floor_velocity(container: ModuleContainer) -> None:
    # vel_deadband attenuates the filtered velocity by eps (soft, continuous):
    # velocity jitter inside the noise floor fully zeroes, so J*acc cannot
    # chatter the min-0 force clamp at near-zero velocity.
    module = _make_module(K_m=0.0, K_e=0.0, K_ms=0.0, K_es=0.0, K_c=1000.0, vel_deadband=0.5)
    _finalize(container, module)

    container.robot.get_tendon_force_buf = torch.zeros((1, 3))
    container.scaled_act_buf = torch.tensor([[-2.0, 2.0, 3.0]], dtype=torch.float32)
    module.step(container)  # first step: no history -> vel = 0 -> acc = 0 -> force = 0
    assert container.tendon_force_buf[:, 0].item() == 0.0

    # Length grows by 0.003: vel_raw = -0.3 -> |vel| < 0.5 -> fully zeroed.
    container.robot.get_tendon_lengths_buf = torch.tensor([[2.003, 3.0, 9.0]])
    module.step(container)
    assert torch.all(module.prev_vel == 0.0)
    # omega = 0 -> acc_raw = 0 -> J*acc = 0 -> force = 0.
    assert container.tendon_force_buf[:, 0].item() == 0.0

    # Length grows by 0.01: vel_raw = -(2.01-2.003)/0.01 = -0.7 -> vel = -0.7 (raw filter)
    # -> soft-debanded to -(0.7 - 0.5) = -0.2 -> omega = -0.2, omega_prev = 0
    # -> acc_raw = -20 -> tau_f_stop = 20 -> tau_e = -40 -> force = 40.
    container.robot.get_tendon_lengths_buf = torch.tensor([[2.01, 3.0, 9.0]])
    module.step(container)
    assert module.prev_vel[0, 0].item() == pytest.approx(-0.2, abs=1e-4)
    assert container.tendon_force_buf[:, 0].item() == pytest.approx(40.0, abs=1e-2)


def test_step_soft_deadband_on_acc(container: ModuleContainer) -> None:
    # acc_deadband attenuates the filtered acceleration by eps (soft, continuous):
    # small alternating accelerations zero out before entering tau_e.
    module = _make_module(K_m=0.0, K_e=0.0, K_ms=0.0, K_es=0.0, K_c=1000.0, acc_deadband=25.0)
    _finalize(container, module)

    container.robot.get_tendon_force_buf = torch.zeros((1, 3))
    container.scaled_act_buf = torch.tensor([[-2.0, 2.0, 3.0]], dtype=torch.float32)
    module.step(container)  # first step: vel = 0 -> acc_raw = 0 -> acc = 0
    assert torch.all(module.prev_acc == 0.0)

    # Length grows by 0.01: vel_raw = -1 -> omega = -1, omega_prev = 0 -> acc_raw = -100
    # -> soft-debanded to -(100 - 25) = -75 -> tau_e = -75 - 100 = -175 -> force = 175.
    container.robot.get_tendon_lengths_buf = torch.tensor([[2.01, 3.0, 9.0]])
    module.step(container)
    assert module.prev_acc[0, 0].item() == pytest.approx(-75.0, rel=1e-5)
    assert container.tendon_force_buf[:, 0].item() == pytest.approx(175.0, rel=1e-5)

    # Constant velocity (length keeps growing by 0.01): acc_raw = 0 -> acc stays 0.
    container.robot.get_tendon_lengths_buf = torch.tensor([[2.02, 3.0, 9.0]])
    module.step(container)
    assert torch.all(module.prev_acc == 0.0)


def test_reset_is_selective_per_env(container: ModuleContainer) -> None:
    container.total_num_envs = 2
    module = _make_module()
    _finalize(container, module)

    container.scaled_act_buf = torch.tensor([[3.0, 3.0], [3.0, 3.0]], dtype=torch.float32)
    container.robot.get_tendon_lengths_buf = torch.tensor([[2.0, 3.0, 9.0], [2.0, 3.0, 9.0]])
    container.robot.get_tendon_force_buf = torch.tensor([[0.5, 0.2, 0.9], [0.5, 0.2, 0.9]])
    module.step(container)

    # Both envs build velocity/acceleration history.
    container.robot.get_tendon_lengths_buf = torch.tensor([[2.01, 3.02, 9.0], [2.01, 3.02, 9.0]])
    module.step(container)

    container.reset_buf = torch.tensor([True, False])
    module.reset(container)

    # Env 0: all history cleared.
    assert torch.all(torch.isnan(module.prev_length[0]))
    assert torch.all(module.prev_vel[0] == 0.0)
    assert torch.all(module.prev_acc[0] == 0.0)
    # Env 1: history preserved.
    assert torch.all(torch.isfinite(module.prev_length[1]))
    assert torch.allclose(module.prev_length[1], torch.tensor([2.01, 3.02]))
    assert module.prev_vel[1, 0].item() != 0.0
    assert module.prev_acc[1, 0].item() != 0.0

    # Env 0's next step uses zero velocity (NaN guard forces vel_raw to 0); env 1 continues from its history.
    container.robot.get_tendon_lengths_buf = torch.tensor([[2.02, 3.04, 9.0], [2.02, 3.04, 9.0]])
    module.step(container)
    assert module.prev_vel[0, 0].item() == 0.0  # fresh: NaN guard keeps the first-step velocity at 0
    assert module.prev_vel[1, 0].item() == pytest.approx(-1.0, abs=1e-4)  # same rate -> same filtered vel


def test_step_spike_guard_rejects_teleport_jumps(container: ModuleContainer) -> None:
    # max_tendon_speed rejects phantom velocities from the reset teleport: a length
    # jump far beyond the physical speed limit zeroes the raw velocity instead of
    # launching the finger.
    module = _make_module(K_m=0.0, K_e=0.0, K_ms=0.0, K_es=0.0, K_c=1000.0, max_tendon_speed=1.5)
    _finalize(container, module)

    container.robot.get_tendon_force_buf = torch.zeros((1, 3))
    container.scaled_act_buf = torch.tensor([[-2.0, 2.0, 3.0]], dtype=torch.float32)
    module.step(container)  # first step: vel = 0 -> force = 0
    assert container.tendon_force_buf[:, 0].item() == 0.0

    # The reset teleport: the length jumps by 0.02 m in one step (vel_raw = -2 >> 1.5).
    # Without the guard: omega = -2 -> acc = -200 -> force = 400 (the launch).
    container.robot.get_tendon_lengths_buf = torch.tensor([[2.02, 3.0, 9.0]])
    module.step(container)
    assert torch.all(module.prev_vel == 0.0)
    assert container.tendon_force_buf[:, 0].item() == 0.0

    # The next step resumes the normal FD from the fresh length.
    container.robot.get_tendon_lengths_buf = torch.tensor([[2.03, 3.0, 9.0]])
    module.step(container)
    assert module.prev_vel[0, 0].item() == pytest.approx(-1.0, abs=1e-4)
    assert container.tendon_force_buf[:, 0].item() == pytest.approx(200.0, rel=1e-5)


def test_finalize_requires_robot(container: ModuleContainer) -> None:
    container.robot = None
    module = _make_module()
    with pytest.raises(RuntimeError, match="requires 'robot'"):
        module.finalize(container)


def test_finalize_requires_env(container: ModuleContainer) -> None:
    container.env = None
    module = _make_module()
    with pytest.raises(RuntimeError, match="requires 'env'"):
        module.finalize(container)


def test_finalize_requires_tendon_robot(container: ModuleContainer) -> None:
    container.robot.use_tendon = False
    module = _make_module()
    with pytest.raises(RuntimeError, match="use_tendon"):
        module.finalize(container)


def test_finalize_validates_model_tendon_indices(container: ModuleContainer) -> None:
    module = _make_module(model_tendon_indices=[])
    with pytest.raises(RuntimeError, match="model_tendon_indices"):
        module.finalize(container)

    module = _make_module(model_tendon_indices=[0, 5])
    with pytest.raises(RuntimeError, match="model_tendon_indices"):
        module.finalize(container)


def test_post_finalize_requires_timestep_and_frame_skip(container: ModuleContainer) -> None:
    del container._data["timestep"]
    module = _make_module()
    module.finalize(container)
    with pytest.raises(RuntimeError, match="'timestep' and 'frame_skip'"):
        module.post_finalize(container)


def test_post_finalize_rejects_nonpositive_pully_radius(container: ModuleContainer) -> None:
    module = _make_module(pully_radius=0.0)
    module.finalize(container)
    with pytest.raises(RuntimeError, match="pully_radius"):
        module.post_finalize(container)


def test_post_finalize_rejects_nonpositive_v_s(container: ModuleContainer) -> None:
    module = _make_module(v_s=0.0)
    module.finalize(container)
    with pytest.raises(RuntimeError, match="v_s"):
        module.post_finalize(container)


def test_post_finalize_rejects_out_of_range_acc_filter_alpha(container: ModuleContainer) -> None:
    module = _make_module(acc_filter_alpha=1.5)
    module.finalize(container)
    with pytest.raises(RuntimeError, match="acc_filter_alpha"):
        module.post_finalize(container)


def test_post_finalize_rejects_out_of_range_vel_filter_alpha(container: ModuleContainer) -> None:
    module = _make_module(vel_filter_alpha=1.5)
    module.finalize(container)
    with pytest.raises(RuntimeError, match="vel_filter_alpha"):
        module.post_finalize(container)


def test_post_finalize_rejects_negative_vel_deadband(container: ModuleContainer) -> None:
    module = _make_module(vel_deadband=-0.1)
    module.finalize(container)
    with pytest.raises(RuntimeError, match="vel_deadband"):
        module.post_finalize(container)


def test_post_finalize_rejects_negative_acc_deadband(container: ModuleContainer) -> None:
    module = _make_module(acc_deadband=-1.0)
    module.finalize(container)
    with pytest.raises(RuntimeError, match="acc_deadband"):
        module.post_finalize(container)

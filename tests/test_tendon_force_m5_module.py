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
        self.get_tendon_vel_buf = torch.zeros((1, num_tendons), dtype=torch.float32)
        self.get_tendon_force_buf = torch.zeros((1, num_tendons), dtype=torch.float32)
        self.calculated_tendon_velocity = torch.zeros((1, num_tendons), dtype=torch.float32)
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
    cont.active_dof_slice = slice(0, 3)
    cont.active_dof_indices = torch.tensor([0, 1, 2])
    cont.scaled_act_buf = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    cont.robot.get_tendon_force_buf = torch.tensor([[0.5, 0.2, 0.9]])
    return cont


def _make_module(**overrides) -> TendonForceM5Module:
    """Module with simple coefficients and radius 1.0 so omega = velocity, acc_motor = acc."""
    config = {
        "model_tendon_indices": [0, 1],
        "pully_radius": 1.0,
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


def _set_signals(container: ModuleContainer, velocity: list[float], acceleration: list[float]) -> None:
    """Feed the calculated tendon signals (tendon units) for the model tendons."""
    robot = container.robot
    robot.calculated_tendon_velocity[:, : len(velocity)] = torch.tensor(velocity, dtype=torch.float32)
    robot.calculated_tendon_acceleration[:, : len(acceleration)] = torch.tensor(acceleration, dtype=torch.float32)


def _finalize(container: ModuleContainer, module: TendonForceM5Module) -> None:
    module.finalize(container)
    module.post_finalize(container)


def test_post_finalize_allocates_buffers(container: ModuleContainer) -> None:
    module = _make_module()
    _finalize(container, module)

    assert container.tendon_force_buf.shape == (1, 3)
    assert torch.equal(container.tendon_model_indices, torch.tensor([0, 1]))
    assert module.dt == pytest.approx(0.01)


def test_step_stick_regime_keeps_previous_force(container: ModuleContainer) -> None:
    # Zero calculated signals: vel = 0, acc = 0, and |tau_m + tau_e_prev| <= tau_f_max:
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


def test_step_uses_calculated_velocity(container: ModuleContainer) -> None:
    # omega = calculated velocity / radius; the simulator velocity buffer is ignored.
    # Huge friction cap (K_c) so tau_f is unclamped: tau_e = J*acc + J/dt*omega + tau_e_prev.
    module = _make_module(K_m=0.0, K_e=0.0, K_ms=0.0, K_es=0.0, K_c=1000.0)
    _finalize(container, module)

    container.robot.get_tendon_vel_buf = torch.full((1, 3), 999.0)  # ignored by the module
    module.step(container)  # zero calculated signals -> vel = 0 -> acc = 0 -> force = sim force
    assert container.tendon_force_buf[:, 0].item() == pytest.approx(0.5, rel=1e-5)
    assert container.tendon_force_buf[:, 1].item() == pytest.approx(0.2, rel=1e-5)

    # Tendon 0: vel = +1 -> omega = 1 -> acc = 100. Tendon 1: vel = -2 -> acc = -200.
    _set_signals(container, [1.0, -2.0], [100.0, -200.0])
    module.step(container)

    # Tendon 0: tau_e = 100 + 100 - 0.5 = 199.5 -> force = -199.5 -> clamped to 0.
    assert container.tendon_force_buf[:, 0].item() == 0.0
    # Tendon 1: tau_e = -200 - 200 - 0.2 = -400.2 -> force = 400.2.
    assert container.tendon_force_buf[:, 1].item() == pytest.approx(400.2, rel=1e-5)


def test_step_positive_force_via_acceleration(container: ModuleContainer) -> None:
    # tau_e = J*acc - tau_m - tau_f with acc from the calculated signals;
    # applied force is -tau_e/radius (vsim convention), so negative tau_e pulls.
    module = _make_module(K_m=0.0, K_e=0.0, K_ms=0.0, K_es=0.0, K_c=1000.0)
    _finalize(container, module)

    container.robot.get_tendon_force_buf = torch.zeros((1, 3))
    container.scaled_act_buf = torch.tensor([[-2.0, 2.0, 3.0]], dtype=torch.float32)
    module.step(container)  # zero signals: vel = 0 -> acc = 0 -> force = 0
    assert container.tendon_force_buf[:, 0].item() == 0.0

    # omega = -1, acc = -100 -> tau_e = -100 - 100 = -200 -> force = 200.
    _set_signals(container, [-1.0], [-100.0])
    module.step(container)
    assert container.tendon_force_buf[:, 0].item() == pytest.approx(200.0, rel=1e-5)

    # omega = 0, acc = +100 -> tau_e = 100 -> force clamped to 0.
    _set_signals(container, [0.0], [100.0])
    module.step(container)
    assert container.tendon_force_buf[:, 0].item() == 0.0


def test_step_converts_signals_via_pully_radius(container: ModuleContainer) -> None:
    # radius 0.5: omega = vel / 0.5 = 2 * vel and acc_motor = acc / 0.5 = 2 * acc,
    # force = -tau_e / 0.5: the same omega/acc as the radius-1 case yields 2x the force.
    module = _make_module(K_m=0.0, K_e=0.0, K_ms=0.0, K_es=0.0, K_c=1000.0, pully_radius=0.5)
    _finalize(container, module)

    container.robot.get_tendon_force_buf = torch.zeros((1, 3))
    container.scaled_act_buf = torch.tensor([[-2.0, 2.0, 3.0]], dtype=torch.float32)
    # vel = -0.5 -> omega = -1; acc = -50 -> acc_motor = -100.
    _set_signals(container, [-0.5], [-50.0])
    module.step(container)
    assert container.tendon_force_buf[:, 0].item() == pytest.approx(400.0, rel=1e-5)


def test_step_supports_multiple_envs(container: ModuleContainer) -> None:
    container.total_num_envs = 2
    module = _make_module()
    _finalize(container, module)

    container.robot.calculated_tendon_velocity = torch.zeros((2, 3))
    container.robot.calculated_tendon_acceleration = torch.zeros((2, 3))
    container.scaled_act_buf = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=torch.float32)
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

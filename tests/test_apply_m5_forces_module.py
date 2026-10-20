"""Unit tests for apply_m5_forces' slider speed limit."""

from types import SimpleNamespace

import pytest
import torch

from optimal_morphology_rl.modules.apply_control.apply_m5_forces_module import ApplyM5ForcesModule
from optimal_morphology_rl.modules.module_container import ModuleContainer

R, J, DT = 0.005, 1.0e-3, 0.005  # pulley radius (m), J_eff (kg m^2), control step (s)


def _module(omega, command, max_slider_velocity=0.0425, tension=0.0, **overrides):
    num_envs = omega.shape[0]
    spool = SimpleNamespace(
        model_tendon_indices=[0],
        dt=DT,
        pully_radius=torch.full((num_envs, 1), R),
        j_eff=torch.full((num_envs, 1), J),
        theta_m=torch.zeros((num_envs, 1)),
        omega_m=omega.reshape(num_envs, 1),
        tension=torch.full((num_envs, 1), tension),
        tau_e=torch.full((num_envs, 1), -R * tension),
        motor_indices=torch.tensor([0]),
        reset_rows=torch.zeros(num_envs, dtype=torch.bool),
        commanded_angle=lambda container: torch.full((num_envs, 1), command),
    )
    container = ModuleContainer()
    container.spool, container.device, container.total_num_envs = spool, torch.device("cpu"), num_envs
    container.robot = SimpleNamespace(num_motors=1)
    # No friction: the servo law is tau = motor_gain * (command - theta).
    config = {"motor_gain": 1.0, "K_v": 0.0, "K_c": 0.0, "K_m": 0.0, "K_e": 0.0, "K_cs": 0.0, "K_ms": 0.0, "K_es": 0.0}
    if max_slider_velocity is not None:
        config["max_slider_velocity"] = max_slider_velocity
    config.update(overrides)
    module = ApplyM5ForcesModule(config)
    module.finalize(container)
    module.post_finalize(container)
    return module, container


def _next_omega(container, omega, tension=0.0):
    """One control step of the loop the simulator closes: J * domega = tau + tau_e."""
    tau = container.spool_motor_force_buf[:, :1] * R
    return omega.reshape(-1, 1) + DT / J * (tau - R * tension)


def test_limit_caps_the_next_speed_at_the_limit() -> None:
    w_max = 0.0425 / R
    omega = torch.tensor([0.0, 0.5 * w_max, w_max, 1.5 * w_max])
    module, container = _module(omega, command=100.0)  # a huge command would accelerate far past the limit
    module.step(container)
    next_omega = _next_omega(container, omega)
    assert torch.all(next_omega <= w_max + 1e-6)
    assert torch.allclose(next_omega[2:], torch.full((2, 1), w_max))  # at / over the limit: held (braked) at it


def test_limit_accounts_for_the_tendon_load() -> None:
    w_max = 0.0425 / R
    omega = torch.tensor([w_max])
    module, container = _module(omega, command=100.0, tension=20.0)
    module.step(container)
    assert torch.allclose(_next_omega(container, omega, tension=20.0), torch.tensor([[w_max]]))


def test_no_effect_below_the_limit_or_without_it() -> None:
    omega = torch.tensor([0.0])
    limited, c1 = _module(omega, command=0.001)
    unlimited, c2 = _module(omega, command=0.001, max_slider_velocity=None)
    limited.step(c1)
    unlimited.step(c2)
    assert torch.equal(c1.spool_motor_force_buf, c2.spool_motor_force_buf)
    assert torch.allclose(c2.spool_motor_force_buf[:, 0], torch.tensor([0.001 / R]))  # tau_m / r


def test_invalid_limit_raises() -> None:
    with pytest.raises(RuntimeError, match="max_slider_velocity"):
        _module(torch.tensor([0.0]), command=0.0, max_slider_velocity=0.0)


def test_action_is_the_desired_spool_angle() -> None:
    # No friction: tau = motor_gain * (target - theta), with motor_gain = 1.
    module, container = _module(torch.zeros(3), command=1.0, max_slider_velocity=None)
    container.spool.theta_m = torch.tensor([[0.0], [0.4], [1.5]])
    module.step(container)
    assert torch.allclose(container.spool_motor_force_buf[:, 0], torch.tensor([1.0, 0.6, -0.5]) / R)


def test_target_is_capped_at_zero() -> None:
    omega = torch.zeros(3)
    module, container = _module(omega, command=-2.0, max_slider_velocity=None)
    container.spool.theta_m = torch.tensor([[0.0], [0.5], [3.0]])  # spool angles; command -2 rad
    module.step(container)
    assert torch.all(module.last_signals["action"] == 0.0)  # max(target, 0)
    assert torch.allclose(container.spool_motor_force_buf[:, 0], torch.tensor([0.0, -0.5, -3.0]) / R)  # (0 - theta) / r


class _ConstantFrictionModel(torch.nn.Module):
    """Friction-model stand-in: constant coefficients (K_c = 1, others 0), same interface as the real one."""

    coefficient_names: list[str]

    def __init__(self):
        super().__init__()
        self.dt = DT
        self.stribeck_velocity = 0.1
        self.stribeck_exponent = 2.0
        self.coefficient_names = ["K_c", "K_v", "K_m", "K_e", "K_cs", "K_ms", "K_es"]
        self.register_buffer("needs_init", torch.ones(1, dtype=torch.bool))
        self.register_buffer("last_coefficients", torch.zeros(1, 7))

    def forward(self, omega, desired, theta, tau_e):
        coefficients = torch.zeros(omega.shape[0], 7)
        coefficients[:, 0] = 1.0
        self.last_coefficients = coefficients
        self.needs_init = torch.zeros(omega.shape[0], dtype=torch.bool)
        return coefficients[:, 0]

    @torch.jit.export
    def reset(self, mask: torch.Tensor | None = None) -> None:
        pass


@pytest.mark.parametrize("gain", [1.0, 0.5])
def test_model_coefficient_gain_scales_the_predicted_friction(tmp_path, gain) -> None:
    path = tmp_path / "friction.pt"
    torch.jit.script(_ConstantFrictionModel()).save(str(path))
    # Stalled spool, command 10 rad: tau_m = 10 Nm against a K_c * gain budget, so tau_f = -K_c * gain.
    module, container = _module(torch.zeros(1), command=10.0, max_slider_velocity=None, friction_model_path=str(path),
                                model_coefficient_gain=gain)
    module.step(container)
    assert module.last_signals["tau_f_max"].item() == pytest.approx(gain)
    assert container.spool_motor_force_buf[0, 0].item() == pytest.approx((10.0 - gain) / R)

"""Unit tests for apply_m5_forces' slider speed limit."""

from types import SimpleNamespace

import pytest
import torch

from optimal_morphology_rl.modules.apply_control.apply_m5_forces_module import ApplyM5ForcesModule
from optimal_morphology_rl.modules.module_container import ModuleContainer

R, J, DT = 0.005, 1.0e-3, 0.005  # pulley radius (m), J_eff (kg m^2), control step (s)


def _module(omega, command, max_spool_velocity=8.5, tension=0.0, **overrides):
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
    # No friction and a unit motor without deadband: tau = command - theta.
    config = {"motor_position_gain": 1.0, "torque_constant": 1.0, "current_deadband": 0.0, "K_v": 0.0, "K_c": 0.0, "K_m": 0.0, "K_e_wind": 0.0, "K_e_unwind": 0.0, "K_cs": 0.0, "K_ms": 0.0, "K_es": 0.0}
    if max_spool_velocity is not None:
        config["max_spool_velocity"] = max_spool_velocity
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
    unlimited, c2 = _module(omega, command=0.001, max_spool_velocity=None)
    limited.step(c1)
    unlimited.step(c2)
    assert torch.equal(c1.spool_motor_force_buf, c2.spool_motor_force_buf)
    assert torch.allclose(c2.spool_motor_force_buf[:, 0], torch.tensor([0.001 / R]))  # tau_m / r


def test_invalid_limit_raises() -> None:
    with pytest.raises(RuntimeError, match="max_spool_velocity"):
        _module(torch.tensor([0.0]), command=0.0, max_spool_velocity=0.0)


def test_action_is_the_change_of_the_spool_angle() -> None:
    # No friction, unit motor: desired = theta + action, so tau = action wherever the spool is.
    module, container = _module(torch.zeros(3), command=1.0, max_spool_velocity=None)
    container.spool.theta_m = torch.tensor([[0.0], [0.4], [1.5]])
    module.step(container)
    assert torch.allclose(module.last_signals["action"][:, 0], torch.tensor([1.0, 1.4, 2.5]))  # the desired spool angle
    assert torch.allclose(container.spool_motor_force_buf[:, 0], torch.ones(3) / R)


def test_desired_angle_is_capped_at_minus_one() -> None:
    omega = torch.zeros(3)
    module, container = _module(omega, command=-2.0, max_spool_velocity=None)
    container.spool.theta_m = torch.tensor([[0.0], [0.5], [3.0]])  # spool angles; action -2 rad
    module.step(container)
    assert torch.allclose(module.last_signals["action"][:, 0], torch.tensor([-1.0, -1.0, 1.0]))  # max(theta - 2, -1)
    assert torch.allclose(container.spool_motor_force_buf[:, 0], torch.tensor([-1.0, -1.5, -2.0]) / R)  # (desired - theta) / r


def test_motor_torque_has_a_current_deadband() -> None:
    # i = 0.1 A/rad * (target - theta); tau = 0.48 Nm/A * sign(i) * max(|i| - 0.042 A, 0).
    module, container = _module(torch.zeros(4), command=0.0, max_spool_velocity=None,
                                motor_position_gain=0.1, torque_constant=0.48, current_deadband=0.042)
    error = torch.tensor([[0.3], [0.42], [1.0], [-1.0]])  # rad: inside, at the edge of and past the deadband
    expected = 0.48 * torch.sign(0.1 * error) * torch.clamp((0.1 * error).abs() - 0.042, min=0.0)
    assert torch.allclose(module._motor_torque(error), expected)
    assert torch.all(module._motor_torque(error)[:2] == 0.0)
    assert module._motor_torque(error)[2].item() == pytest.approx(0.48 * 0.058)
    assert module._motor_torque(error)[3].item() == pytest.approx(-0.48 * 0.058)


class _EnvelopeModel(torch.nn.Module):
    """Envelope-model stand-in with the real interface: tau_f_max = 0.5 + |theta_delta| (Nm), per-row history."""

    def __init__(self):
        super().__init__()
        self.frequency = 200
        self.register_buffer("force_buffer", torch.zeros(0, 1))
        self.register_buffer("resets", torch.zeros(0))

    def forward(self, omega: torch.Tensor, theta_delta: torch.Tensor) -> torch.Tensor:
        if self.force_buffer.shape[0] != omega.shape[0]:
            self.force_buffer = torch.zeros(omega.shape[0], 1)
            self.resets = torch.zeros(omega.shape[0])
        return 0.5 + theta_delta.abs()

    @torch.jit.export
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.resets += 1.0
        else:
            self.resets[env_ids] += 1.0


def _with_envelope_model(tmp_path, **overrides):
    path = tmp_path / "envelope.pt"
    torch.jit.script(_EnvelopeModel()).save(str(path))
    return _module(torch.zeros(2), command=1.0, max_spool_velocity=None, friction_model_path=str(path), **overrides)


@pytest.mark.parametrize("gain", [1.0, 0.5, 0.0])
def test_friction_model_envelope_is_the_budget(tmp_path, gain) -> None:
    # Stalled spools, action 1 rad (theta_delta = desired - theta = 1, theta 0 / 0.4): envelope 0.5 + |theta_delta|
    # (x the curriculum factor); unit motor tau = theta_delta.
    module, container = _with_envelope_model(tmp_path)
    module.set_curriculum_factor(gain, container)
    container.spool.theta_m = torch.tensor([[0.0], [0.4]])
    module.step(container)
    envelope = gain * torch.tensor([1.5, 1.5])
    assert torch.allclose(module.last_signals["tau_f_max"][:, 0], envelope)
    # The stall torque tau_m (1.0) is held up to the envelope.
    tau = torch.ones(2) - torch.minimum(torch.ones(2), envelope)
    assert torch.allclose(container.spool_motor_force_buf[:, 0], tau / R)


def test_friction_scale_and_reset_rows(tmp_path) -> None:
    module, container = _with_envelope_model(tmp_path)
    module.step(container)  # allocates the model's per-row state
    module.friction_scale[:] = torch.tensor([[2.0], [1.0]])  # per env
    module.step(container)
    assert torch.allclose(module.last_signals["tau_f_max"][:, 0], torch.tensor([3.0, 1.5]))
    container.reset_buf = torch.tensor([False, True])
    module.reset(container)
    assert module.friction_model.resets.tolist() == [0.0, 1.0]  # only the reset env's row restarts


def test_friction_model_reset_before_the_first_step_is_skipped(tmp_path) -> None:
    module, container = _with_envelope_model(tmp_path)
    container.reset_buf = torch.ones(2, dtype=torch.bool)
    module.reset(container)  # no per-row state yet: nothing to restart
    module.step(container)
    assert module.friction_model.resets.tolist() == [0.0, 0.0]


@pytest.mark.parametrize("omega, tau_m, tau_e, k_e", [
    (1.0, 0.0, -0.1, 0.12),    # winding (theta increasing)
    (-1.0, 0.0, -0.1, 0.41),   # unwinding
    (0.0, 0.2, -0.1, 0.12),    # at rest, pushed to wind (tau_m + tau_e > 0)
    (0.0, 0.05, -0.1, 0.41),   # at rest, the tendon pulls it to unwind
])
def test_load_friction_depends_on_the_winding_direction(omega, tau_m, tau_e, k_e) -> None:
    # Only the load term: budget = K_c + |K_m tau_m - K_e tau_e| with K_c = K_m = 0 -> K_e * |tau_e|.
    module, _ = _module(torch.zeros(1), command=0.0, max_spool_velocity=None, K_e_wind=0.12, K_e_unwind=0.41)
    budget = module._friction_budget(torch.tensor([[tau_m]]), torch.tensor([[tau_e]]), torch.tensor([[omega]]))
    assert budget.item() == pytest.approx(k_e * abs(tau_e))


def test_friction_model_torque_output_is_applied_as_is(tmp_path) -> None:
    # The stand-in returns 0.5 + |theta_delta|; as a torque model that is tau_f itself (no stop torque, no clamp).
    module, container = _with_envelope_model(tmp_path, friction_model_output="torque")
    module.set_curriculum_factor(0.5, container)
    container.spool.theta_m = torch.tensor([[0.0], [0.4]])
    module.step(container)
    tau_f = 0.5 * torch.tensor([1.5, 1.5])  # theta_delta = the action, 1 rad
    assert torch.allclose(module.last_signals["tau_f"][:, 0], tau_f)
    assert torch.allclose(container.spool_motor_force_buf[:, 0], (torch.ones(2) + tau_f) / R)


def test_invalid_friction_model_output_raises(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="friction_model_output"):
        _with_envelope_model(tmp_path, friction_model_output="budget")


def test_curriculum_factor_scales_the_m5_friction() -> None:
    # Stalled spool, unit motor: tau_m = 1 against a K_c budget of 0.5 x factor.
    for factor, held in ((1.0, 0.5), (0.5, 0.25), (0.0, 0.0)):
        module, container = _module(torch.zeros(1), command=1.0, max_spool_velocity=None, K_c=0.5)
        module.set_curriculum_factor(factor, container)
        module.step(container)
        assert container.spool_motor_force_buf[0, 0].item() == pytest.approx((1.0 - held) / R)


def test_friction_scale_scales_the_m5_budget_per_env() -> None:
    # Stalled spools, unit motor: tau_m = 1 against a K_c budget of 0.5 x the env's friction scale.
    module, container = _module(torch.zeros(2), command=1.0, max_spool_velocity=None, K_c=0.5)
    module.friction_scale[:] = torch.tensor([[1.5], [0.5]])
    module.step(container)
    assert torch.allclose(container.spool_motor_force_buf[:, 0], torch.tensor([1.0 - 0.75, 1.0 - 0.25]) / R)

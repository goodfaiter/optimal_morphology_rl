"""Unit tests for the tendon_est module."""

from pathlib import Path

import pytest
import torch

from optimal_morphology_rl.modules import tendon_est_module
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.tendon_est_module import TendonEstModule


class _TinySpringModel(torch.nn.Module):
    """Inner stateful model: deterministic sums of input columns plus a growing shift."""

    shift: torch.Tensor

    def __init__(self):
        super().__init__()
        self.register_buffer("shift", torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = x[:, 0, :] + self.shift
        self.shift.add_(1.0)
        force = values.sum(dim=-1, keepdim=True)
        spring = torch.ones_like(force)
        return torch.cat([force, spring], dim=-1).unsqueeze(1)

    @torch.jit.export
    def reset(self, mask: torch.Tensor) -> None:
        # The broadcast state resets whenever any row is selected.
        if torch.count_nonzero(mask.float()).item() > 0:
            self.shift = torch.zeros_like(self.shift)


class _TinySpringWrapper(torch.nn.Module):
    """Tiny spring-transformer lookalike with the same contract as the real checkpoint."""

    metadata: dict[str, int]

    def __init__(self):
        super().__init__()
        self.register_buffer("input_mean", torch.zeros(1, 3))
        self.register_buffer("input_std", torch.ones(1, 3))
        self.register_buffer("output_mean", torch.zeros(1, 2))
        self.register_buffer("output_std", torch.ones(1, 2))
        self.model_type = "SpringTransformerModel"
        self.metadata = {"frequency": 200, "history_size": 50, "stride": 2}
        self.input_columns = [
            "measured_position_rad_data",
            "desired_position_rad_data",
            "measured_velocity_rad_per_sec_data",
        ]
        self.output_columns = ["tendon_bota_force_newton_data", "spring_coeff"]
        self.model = _TinySpringModel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.input_mean) / self.input_std
        out = self.model(x)
        return out * self.output_std + self.output_mean


class _FakeEnv:
    pass


class _FakeRobot:
    def __init__(self, num_tendons: int = 3):
        self.use_tendon = True
        self.num_tendons = num_tendons
        self.get_tendon_lengths_buf = torch.zeros((1, num_tendons), dtype=torch.float32)
        self.get_tendon_vel_buf = torch.zeros((1, num_tendons), dtype=torch.float32)


def _make_checkpoint(tmp_path: Path) -> str:
    model_path = str(tmp_path / "tiny_spring_model.pt")
    torch.jit.script(_TinySpringWrapper()).save(model_path)
    return model_path


@pytest.fixture
def container() -> ModuleContainer:
    cont = ModuleContainer()
    cont.total_num_envs = 1
    cont.device = torch.device("cpu")
    cont.env = _FakeEnv()
    cont.robot = _FakeRobot()
    cont.active_dof_slice = slice(0, 3)
    cont.active_dof_indices = torch.tensor([0, 1, 2])
    cont.scaled_act_buf = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    cont.reset_buf = torch.zeros(1, dtype=torch.bool)
    cont.robot.get_tendon_lengths_buf = torch.tensor([[0.066, 0.055, 0.07]])
    cont.robot.get_tendon_vel_buf = torch.tensor([[0.011, 0.0, 0.0]])
    return cont


def _make_module(model_path: str, **overrides) -> TendonEstModule:
    config = {
        "model_path": model_path,
        "pully_radius": 0.011,
        "model_tendon_indices": [0, 1],
        "zero_offset_length": [0.0, 0.0],
        "desired_min": -10.0,  # wide clamps so the step math is exact
        "desired_max": 10.0,
        "force_min": -100.0,
        "force_max": 100.0,
    }
    config.update(overrides)
    return TendonEstModule(config)


def test_post_finalize_loads_single_runner_and_allocates(container: ModuleContainer, tmp_path: Path) -> None:
    module = _make_module(_make_checkpoint(tmp_path))
    module.finalize(container)
    module.post_finalize(container)

    # One model, num_envs = total_num_envs x num_model_tendons.
    assert module.model.num_envs == 2
    assert module.force_output_idx == 0
    assert container.tendon_force_buf.shape == (1, 3)
    assert torch.equal(container.tendon_model_indices, torch.tensor([0, 1]))


def test_step_writes_model_forces_for_model_tendons_only(container: ModuleContainer, tmp_path: Path) -> None:
    module = _make_module(_make_checkpoint(tmp_path))
    module.finalize(container)
    module.post_finalize(container)
    module.step(container)

    # measured = -length / radius; desired = clamp(measured + action); vel = -v / radius.
    # module force = -1 * tiny-model output (first step, shift 0) = -(measured + desired + vel).
    measured_0 = -0.066 / 0.011
    measured_1 = -0.055 / 0.011
    expected_0 = -(measured_0 + (measured_0 + 1.0) + -1.0)
    expected_1 = -(measured_1 + (measured_1 + 2.0) + 0.0)

    assert container.tendon_force_buf[:, 0].item() == pytest.approx(expected_0, rel=1e-5)
    assert container.tendon_force_buf[:, 1].item() == pytest.approx(expected_1, rel=1e-5)
    # The DIP tendon (last column) is never written by the module.
    assert container.tendon_force_buf[:, 2].item() == 0.0


def test_step_runs_single_flattened_forward(container: ModuleContainer, tmp_path: Path) -> None:
    module = _make_module(_make_checkpoint(tmp_path))
    module.finalize(container)
    module.post_finalize(container)
    module.step(container)
    first = container.tendon_force_buf.clone()

    # The tiny state grows per forward pass: the summed shift changes the force
    # by 3 per step (each of the 3 features gets the shift before summing).
    module.step(container)
    assert torch.allclose(container.tendon_force_buf[:, :2], first[:, :2] - 3.0)


def test_step_supports_multiple_envs(container: ModuleContainer, tmp_path: Path) -> None:
    # Restriction lifted: batched rows = 2 envs x 2 model tendons, single forward.
    container.total_num_envs = 2
    module = _make_module(_make_checkpoint(tmp_path))
    module.finalize(container)
    module.post_finalize(container)

    assert module.model.num_envs == 4

    container.scaled_act_buf = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    container.robot.get_tendon_lengths_buf = torch.tensor([[0.066, 0.055, 0.07], [0.066, 0.055, 0.07]])
    container.robot.get_tendon_vel_buf = torch.tensor([[0.011, 0.0, 0.0], [0.011, 0.0, 0.0]])
    module.step(container)

    measured_0 = -0.066 / 0.011
    measured_1 = -0.055 / 0.011
    expected_0 = -(measured_0 + (measured_0 + 1.0) + -1.0)
    expected_1 = -(measured_1 + (measured_1 + 2.0) + 0.0)

    # Env-major rows: env 0 and env 1 produce identical rows for identical inputs.
    assert torch.allclose(container.tendon_force_buf[:, 0], torch.full((2,), expected_0, dtype=torch.float32))
    assert torch.allclose(container.tendon_force_buf[:, 1], torch.full((2,), expected_1, dtype=torch.float32))
    assert torch.all(container.tendon_force_buf[:, 2] == 0.0)


def test_reset_clears_model_state(container: ModuleContainer, tmp_path: Path) -> None:
    module = _make_module(_make_checkpoint(tmp_path))
    module.finalize(container)
    module.post_finalize(container)

    module.step(container)
    first = container.tendon_force_buf.clone()
    container.reset_buf[0] = True
    module.reset(container)
    module.step(container)

    assert torch.allclose(container.tendon_force_buf[:, :2], first[:, :2])


def test_step_clamps_predicted_forces(container: ModuleContainer, tmp_path: Path) -> None:
    module = _make_module(_make_checkpoint(tmp_path), force_max=-9.0)
    module.finalize(container)
    module.post_finalize(container)

    module.step(container)
    # Raw outputs (-12, -8) are negated (+12, +8) and clamped to force_max=-9.
    assert container.tendon_force_buf[:, 0].item() == pytest.approx(-9.0)
    assert container.tendon_force_buf[:, 1].item() == pytest.approx(-9.0)


def test_default_config_clamps(container: ModuleContainer, tmp_path: Path) -> None:
    module = TendonEstModule({
        "model_path": _make_checkpoint(tmp_path),
        "pully_radius": 0.011,
        "model_tendon_indices": [0, 1],
        "zero_offset_length": [0.0, 0.0],
    })
    module.finalize(container)
    module.post_finalize(container)

    module.step(container)
    # Defaults: desired clamped to [0, 2*pi]; negated outputs (7, 5) stay within [0, 30].
    assert container.tendon_force_buf[:, 0].item() == pytest.approx(7.0, rel=1e-5)
    assert container.tendon_force_buf[:, 1].item() == pytest.approx(5.0, rel=1e-5)


def test_finalize_requires_robot(container: ModuleContainer, tmp_path: Path) -> None:
    container.robot = None
    module = _make_module(_make_checkpoint(tmp_path))
    with pytest.raises(RuntimeError, match="requires 'robot'"):
        module.finalize(container)


def test_finalize_requires_env(container: ModuleContainer, tmp_path: Path) -> None:
    container.env = None
    module = _make_module(_make_checkpoint(tmp_path))
    with pytest.raises(RuntimeError, match="requires 'env'"):
        module.finalize(container)


def test_finalize_requires_tendon_robot(container: ModuleContainer, tmp_path: Path) -> None:
    container.robot.use_tendon = False
    module = _make_module(_make_checkpoint(tmp_path))
    with pytest.raises(RuntimeError, match="use_tendon"):
        module.finalize(container)


def test_finalize_validates_model_tendon_indices(container: ModuleContainer, tmp_path: Path) -> None:
    module = _make_module(_make_checkpoint(tmp_path), model_tendon_indices=[])
    with pytest.raises(RuntimeError, match="model_tendon_indices"):
        module.finalize(container)

    module = _make_module(_make_checkpoint(tmp_path), model_tendon_indices=[0, 5])
    with pytest.raises(RuntimeError, match="model_tendon_indices"):
        module.finalize(container)


def test_post_finalize_requires_zero_offset_length(container: ModuleContainer, tmp_path: Path) -> None:
    module = TendonEstModule({
        "model_path": _make_checkpoint(tmp_path),
        "pully_radius": 0.011,
        "model_tendon_indices": [0, 1],
    })
    module.finalize(container)
    with pytest.raises(RuntimeError, match="zero_offset_length"):
        module.post_finalize(container)


def test_post_finalize_rejects_wrong_zero_offset_length(container: ModuleContainer, tmp_path: Path) -> None:
    module = _make_module(_make_checkpoint(tmp_path), zero_offset_length=[0.0])
    module.finalize(container)
    with pytest.raises(RuntimeError, match="zero_offset_length"):
        module.post_finalize(container)


def test_step_maps_zero_offsets_per_tendon(container: ModuleContainer, tmp_path: Path) -> None:
    # Zero offsets apply to the model tendons only (tendon 0 and tendon 1).
    module = _make_module(_make_checkpoint(tmp_path), zero_offset_length=[0.011, 0.0])
    module.finalize(container)
    module.post_finalize(container)
    module.step(container)

    # measured = -(length - offset) / radius; module force = -1 * tiny-model output.
    # Tendon 0: measured = -(0.066 - 0.011)/0.011 = -5.0, desired = -4.0, vel = -1.0 -> output -10 -> force 10.
    assert container.tendon_force_buf[:, 0].item() == pytest.approx(10.0, rel=1e-5)
    # Tendon 1: offset 0.0 -> unchanged from the zero-offset base case (output -8 -> force 8).
    assert container.tendon_force_buf[:, 1].item() == pytest.approx(8.0, rel=1e-5)
    # Tendon 2 is never written by the module.
    assert container.tendon_force_buf[:, 2].item() == 0.0


def test_post_finalize_requires_existing_model(container: ModuleContainer, tmp_path: Path) -> None:
    module = _make_module(str(tmp_path / "missing_model.pt"))
    module.finalize(container)
    with pytest.raises(FileNotFoundError, match="model not found"):
        module.post_finalize(container)


def test_step_maps_action_slots_per_tendon(container: ModuleContainer, tmp_path: Path) -> None:
    # Fixed tendon 0 excluded from the policy: active dofs [1, 2], model tendons [1, 2].
    module = _make_module(_make_checkpoint(tmp_path), model_tendon_indices=[1, 2])
    module.finalize(container)
    module.post_finalize(container)

    container.active_dof_indices = torch.tensor([1, 2])
    container.active_dof_slice = slice(0, 2)
    container.scaled_act_buf = torch.tensor([[1.0, 2.0]])
    module.step(container)

    # desired = clamp(measured + actions[:, slot]); slot 0 belongs to tendon 1.
    # Tiny model force (first step, shift 0) = -(measured + desired + vel).
    measured_1 = -0.055 / 0.011
    measured_2 = -0.07 / 0.011
    expected_1 = -(measured_1 + (measured_1 + 1.0) + 0.0)   # slot 0 action (1.0)
    expected_2 = -(measured_2 + (measured_2 + 2.0) + 0.0)   # slot 1 action (2.0)

    assert container.tendon_force_buf[:, 1].item() == pytest.approx(expected_1, rel=1e-5)
    assert container.tendon_force_buf[:, 2].item() == pytest.approx(expected_2, rel=1e-5)
    # Tendon 0 is never written by the module.
    assert container.tendon_force_buf[:, 0].item() == 0.0


class _FakeRunner:
    """Captures construction and reset calls (used for reset-mask expansion)."""

    def __init__(self, model_path: str, device: str = "cpu", num_envs: int = 1):
        self.model_path = model_path
        self.device = device
        self.num_envs = num_envs
        self.reset_masks: list[torch.Tensor] = []
        self.input_columns = [
            "measured_position_rad_data",
            "desired_position_rad_data",
            "measured_velocity_rad_per_sec_data",
        ]
        self.output_columns = ["tendon_bota_force_newton_data", "spring_coeff"]

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        rows = timestep.size(0)
        return torch.zeros((rows, 1, 2))

    def reset_idx(self, mask: torch.Tensor) -> None:
        self.reset_masks.append(mask)


def test_reset_expands_mask_per_model_tendon(container: ModuleContainer, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(tendon_est_module, "ModelRunner", _FakeRunner)
    container.total_num_envs = 2  # 2 envs x 2 model tendons = 4 rows
    module = _make_module(_make_checkpoint(tmp_path))
    module.finalize(container)
    module.post_finalize(container)

    # 2 envs x 2 model tendons = 4 rows.
    assert module.model.num_envs == 4

    container.reset_buf = torch.tensor([True, False])
    module.reset(container)

    # Env-major: env 0 owns rows [0, 2), env 1 owns rows [2, 4).
    assert torch.equal(module.model.reset_masks[-1], torch.tensor([True, True, False, False]))

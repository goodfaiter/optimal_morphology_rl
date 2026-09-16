"""Unit tests for the tendon_est module."""

from pathlib import Path

import pytest
import torch

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
        keep = torch.ones_like(self.shift) * mask.logical_not().float()
        self.shift = self.shift * keep


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
    cont.active_motor_slice = slice(0, 3)
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
        "desired_min": -10.0,  # wide clamps so the step math is exact
        "desired_max": 10.0,
        "force_min": -100.0,
        "force_max": 100.0,
    }
    config.update(overrides)
    return TendonEstModule(config)


def test_post_finalize_loads_runners_and_allocates(container: ModuleContainer, tmp_path: Path) -> None:
    module = _make_module(_make_checkpoint(tmp_path))
    module.finalize(container)
    module.post_finalize(container)

    assert len(module.runners) == 2
    assert module.force_output_idx == 0
    assert container.tendon_force_buf.shape == (1, 3)
    assert torch.allclose(container.tendon_model_indices, torch.tensor([0, 1]))


def test_step_writes_model_forces_for_model_tendons_only(container: ModuleContainer, tmp_path: Path) -> None:
    module = _make_module(_make_checkpoint(tmp_path))
    module.finalize(container)
    module.post_finalize(container)
    module.step(container)

    # measured = -length / radius; desired = clamp(measured + action); vel = -v / radius.
    # Tiny model force (first step, shift 0) = measured + desired + vel.
    measured_0 = -0.066 / 0.011
    measured_1 = -0.055 / 0.011
    expected_0 = measured_0 + (measured_0 + 1.0) + -1.0
    expected_1 = measured_1 + (measured_1 + 2.0) + 0.0

    assert container.tendon_force_buf[:, 0].item() == pytest.approx(expected_0, rel=1e-5)
    assert container.tendon_force_buf[:, 1].item() == pytest.approx(expected_1, rel=1e-5)
    # The DIP tendon (last column) is never written by the module.
    assert container.tendon_force_buf[:, 2].item() == 0.0


def test_step_tracks_stateful_model_tendons(container: ModuleContainer, tmp_path: Path) -> None:
    module = _make_module(_make_checkpoint(tmp_path))
    module.finalize(container)
    module.post_finalize(container)

    module.step(container)
    first = container.tendon_force_buf.clone()
    module.step(container)
    # Tiny inner model adds the shift to each input column per forward (sum grows by 3).
    assert torch.allclose(container.tendon_force_buf[:, :2], first[:, :2] + 3.0)


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
    # Tendon 0 force (-12) is within [-100, -9]; tendon 1 (-8) is clamped up to -9.
    assert container.tendon_force_buf[:, 0].item() == pytest.approx(-12.0, rel=1e-5)
    assert container.tendon_force_buf[:, 1].item() == pytest.approx(-9.0)


def test_default_config_clamps(container: ModuleContainer, tmp_path: Path) -> None:
    module = TendonEstModule({
        "model_path": _make_checkpoint(tmp_path),
        "pully_radius": 0.011,
        "model_tendon_indices": [0, 1],
    })
    module.finalize(container)
    module.post_finalize(container)

    module.step(container)
    # Defaults: desired clamped to [0, 2*pi], force clamped to [0, 30] -> both negatives clamp to 0.
    assert container.tendon_force_buf[:, 0].item() == 0.0
    assert container.tendon_force_buf[:, 1].item() == 0.0


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


def test_finalize_requires_single_env(container: ModuleContainer, tmp_path: Path) -> None:
    container.total_num_envs = 2
    module = _make_module(_make_checkpoint(tmp_path))
    with pytest.raises(RuntimeError, match="single environment"):
        module.finalize(container)


def test_finalize_validates_model_tendon_indices(container: ModuleContainer, tmp_path: Path) -> None:
    module = _make_module(_make_checkpoint(tmp_path), model_tendon_indices=[])
    with pytest.raises(RuntimeError, match="model_tendon_indices"):
        module.finalize(container)

    module = _make_module(_make_checkpoint(tmp_path), model_tendon_indices=[0, 5])
    with pytest.raises(RuntimeError, match="model_tendon_indices"):
        module.finalize(container)


def test_post_finalize_requires_existing_model(container: ModuleContainer, tmp_path: Path) -> None:
    module = _make_module(str(tmp_path / "missing_model.pt"))
    module.finalize(container)
    with pytest.raises(FileNotFoundError, match="model not found"):
        module.post_finalize(container)

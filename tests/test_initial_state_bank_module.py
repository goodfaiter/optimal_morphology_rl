"""Unit tests for initial_state_bank (random initial poses from recorded joint states)."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from optimal_morphology_rl.modules.initial_state_bank_module import InitialStateBankModule, resolve_states_path
from optimal_morphology_rl.modules.module_container import ModuleContainer

NAMES = ["MCP", "PIP", "slider"]


class _Gym:
    def __init__(self):
        self.applied = 0

    def set_articulation_kinematic_states(self, command):
        self.applied += 1


def _states_file(tmp_path: Path, states, names=NAMES) -> Path:
    path = tmp_path / "states.npz"
    np.savez(path, joint_pos=np.asarray(states, dtype=np.float32), joint_names=np.array(names))
    return path


def _module(tmp_path: Path, prob=1.0, states=((0.5, 0.4, 0.01), (1.0, 0.8, 0.02)), names=NAMES, num_envs=4):
    container = ModuleContainer()
    container.device = torch.device("cpu")
    container.robot = SimpleNamespace(art_def=SimpleNamespace(get_joint_dof_def_names=lambda: list(NAMES)))
    container.gym = _Gym()
    container.gpu_reset_kinematic_state_command_array = object()
    container.reset_joint_pos_buf = torch.zeros((num_envs, 3))
    container.reset_joint_vel_buf = torch.ones((num_envs, 3))
    container.reset_buf = torch.ones(num_envs, dtype=torch.bool)
    module = InitialStateBankModule({"path": str(_states_file(tmp_path, states, names)), "prob": prob})
    module.finalize(container)
    module.post_finalize(container)
    return module, container


def test_resets_start_from_stored_states_at_zero_velocity(tmp_path: Path) -> None:
    module, container = _module(tmp_path, prob=1.0)
    module.reset(container)
    stored = module.states
    for row in container.reset_joint_pos_buf:
        assert any(torch.equal(row, s) for s in stored)
    assert torch.all(container.reset_joint_vel_buf == 0.0)
    assert container.gym.applied == 1  # create_robot's reset command re-applied with the stored rows


def test_only_resetting_envs_are_changed(tmp_path: Path) -> None:
    module, container = _module(tmp_path, prob=1.0)
    container.reset_buf[:] = torch.tensor([False, True, False, False])
    module.reset(container)
    assert torch.all(container.reset_joint_pos_buf[[0, 2, 3]] == 0.0)
    assert container.reset_joint_pos_buf[1].abs().sum() > 0


def test_prob_mixes_stored_and_zero_starts(tmp_path: Path) -> None:
    module, container = _module(tmp_path, prob=0.5, num_envs=2000)
    torch.manual_seed(0)
    module.reset(container)
    share = (container.reset_joint_pos_buf.abs().sum(1) > 0).float().mean().item()
    assert 0.45 < share < 0.55


def test_prob_zero_is_a_noop(tmp_path: Path) -> None:
    module, container = _module(tmp_path, prob=0.0)
    module.reset(container)
    assert torch.all(container.reset_joint_pos_buf == 0.0) and container.gym.applied == 0


def test_states_recorded_for_other_joints_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="was recorded for joints"):
        _module(tmp_path, names=["MCP", "PIP", "other"])


def test_missing_file_and_config_errors(tmp_path: Path) -> None:
    container = ModuleContainer()
    container.robot = object()
    module = InitialStateBankModule({"path": str(tmp_path / "nope.npz")})
    with pytest.raises(RuntimeError, match="not found"):
        module.post_finalize(container)
    with pytest.raises(RuntimeError, match="requires 'path'"):
        InitialStateBankModule({})
    with pytest.raises(RuntimeError, match="prob"):
        InitialStateBankModule({"path": "x.npz", "prob": 1.5})


def test_relative_paths_resolve_against_the_envs_folder() -> None:
    path = resolve_states_path("adapt_finger/initial_states.npz")
    assert path.parts[-3:] == ("envs", "adapt_finger", "initial_states.npz")
    assert path.is_file()  # the recorded states shipped with the task

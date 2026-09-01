"""Unit tests for the process_actions module."""

from __future__ import annotations

import pytest
import torch
from vlearn.spaces import Box

from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.process_actions_module import ProcessActionsModule


class _FakeEnv:
    def __init__(self, num_actions: int):
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(num_actions,),
            dtype=float,
        )


class _FakeRobot:
    def __init__(self, num_actions: int, fixed_hand: bool = False):
        self.fixed_hand = fixed_hand
        self.root_slice = slice(0, 0) if fixed_hand else slice(0, 6)
        self.dof_slice = slice(0, num_actions) if fixed_hand else slice(6, 6 + num_actions - 6)
        num_dofs = num_actions if fixed_hand else num_actions - 6
        device = torch.device("cpu")
        self.velocity_scale = torch.tensor([1.0, 1.0, 1.0, 0.2, 0.2, 0.2], dtype=torch.float32, device=device)
        self.max_velocity = self.velocity_scale * 2.0
        self.min_revolute_scale = torch.full((num_dofs,), -0.1, dtype=torch.float32, device=device)
        self.max_revolute_scale = torch.full((num_dofs,), 0.1, dtype=torch.float32, device=device)
        self.scaled_act_buf = None


@pytest.fixture
def container() -> ModuleContainer:
    cont = ModuleContainer()
    cont.total_num_envs = 4
    cont.device = torch.device("cpu")
    cont.env = _FakeEnv(num_actions=8)
    cont.robot = _FakeRobot(num_actions=8, fixed_hand=False)
    cont.reset_buf = torch.zeros(4, dtype=torch.bool)
    return cont


def test_post_finalize_allocates_buffers(container: ModuleContainer) -> None:
    module = ProcessActionsModule({})
    module.finalize(container)
    module.post_finalize(container)

    assert container.actions is not None
    assert container.act_buf is not None
    assert container.last_act_buf is not None
    assert container.scaled_act_buf is not None
    assert container.robot.scaled_act_buf is container.scaled_act_buf
    assert container.actions.shape == (4, 8)
    assert container.act_buf.shape == (4, 8)
    assert container.last_act_buf.shape == (4, 8)
    assert container.scaled_act_buf.shape == (4, 8)


def test_finalize_requires_robot(container: ModuleContainer) -> None:
    container.robot = None
    module = ProcessActionsModule({})
    with pytest.raises(RuntimeError, match="requires 'robot'"):
        module.finalize(container)


def test_finalize_requires_env(container: ModuleContainer) -> None:
    container.env = None
    module = ProcessActionsModule({})
    with pytest.raises(RuntimeError, match="requires 'env'"):
        module.finalize(container)


def test_post_finalize_requires_action_space(container: ModuleContainer) -> None:
    container.env.action_space = None  # type: ignore[assignment]
    module = ProcessActionsModule({})
    module.finalize(container)
    with pytest.raises(RuntimeError, match="action_space"):
        module.post_finalize(container)


def test_step_updates_buffers(container: ModuleContainer) -> None:
    module = ProcessActionsModule({})
    module.finalize(container)
    module.post_finalize(container)

    container.act_buf[:] = torch.arange(8, dtype=torch.float32)
    new_actions = torch.ones((4, 8), dtype=torch.float32) * 0.5
    container.actions[:] = new_actions

    module.step(container)

    assert torch.allclose(container.last_act_buf, torch.arange(8, dtype=torch.float32))
    assert torch.allclose(container.act_buf, new_actions)

    # Scaling: root DOFs scaled by velocity_scale, joint DOFs by revolute scale.
    expected_root = container.robot.velocity_scale[:6] * 0.5
    expected_dof = container.robot.max_revolute_scale * 0.5
    assert torch.allclose(container.scaled_act_buf[:, :6], expected_root)
    assert torch.allclose(container.scaled_act_buf[:, 6:], expected_dof)


def test_step_preserves_per_env_actions(container: ModuleContainer) -> None:
    module = ProcessActionsModule({})
    module.finalize(container)
    module.post_finalize(container)

    new_actions = torch.tensor(
        [
            [0.0] * 8,
            [0.25] * 8,
            [0.5] * 8,
            [1.0] * 8,
        ],
        dtype=torch.float32,
    )
    container.actions[:] = new_actions
    module.step(container)

    assert torch.allclose(container.act_buf, new_actions)
    ratios = [0.0, 0.25, 0.5, 1.0]
    for i, ratio in enumerate(ratios):
        assert torch.allclose(container.scaled_act_buf[i], container.scaled_act_buf[3] * ratio)


def test_reset_zeros_buffers(container: ModuleContainer) -> None:
    module = ProcessActionsModule({})
    module.finalize(container)
    module.post_finalize(container)

    container.act_buf[:] = 1.0
    container.last_act_buf[:] = 2.0
    container.scaled_act_buf[:] = 3.0
    container.reset_buf[[1, 3]] = True

    module.reset(container)

    assert torch.all(container.act_buf[0] != 0.0)
    assert torch.all(container.act_buf[1] == 0.0)
    assert torch.all(container.act_buf[2] != 0.0)
    assert torch.all(container.act_buf[3] == 0.0)

    assert torch.all(container.last_act_buf[0] != 0.0)
    assert torch.all(container.last_act_buf[1] == 0.0)
    assert torch.all(container.last_act_buf[2] != 0.0)
    assert torch.all(container.last_act_buf[3] == 0.0)

    assert torch.all(container.scaled_act_buf[0] != 0.0)
    assert torch.all(container.scaled_act_buf[1] == 0.0)
    assert torch.all(container.scaled_act_buf[2] != 0.0)
    assert torch.all(container.scaled_act_buf[3] == 0.0)

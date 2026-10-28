"""Unit tests for the training curriculum: rl_games RewardPlateauCurriculum -> VlearnEnv -> modules."""

from types import SimpleNamespace

import pytest
from rl_games.common.curriculum import RewardPlateauCurriculum

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_manager import ModuleManager
from optimal_morphology_rl.runners.modular_runner import VlearnEnv


def _curriculum(**overrides):
    config = {"levels": [0.0, 0.5, 1.0], "patience": 3, "min_delta": 0.01, "min_epochs": 2}
    config.update(overrides)
    return RewardPlateauCurriculum(config)


def test_curriculum_advances_on_each_plateau_and_stops_at_the_last_level() -> None:
    curriculum = _curriculum()
    assert curriculum.factor == 0.0 and not curriculum.finished
    events = []
    for epoch in range(1, 30):
        step = curriculum.update(epoch, 10.0)  # flat reward: plateaus
        if step:
            events.append((epoch, step))
        if step == curriculum.STOP:
            break
    # Per level: 2 warm-up epochs not counted, best at the next one, + patience 3; the last level's plateau stops.
    assert events == [(5, curriculum.NEXT_LEVEL), (10, curriculum.NEXT_LEVEL), (15, curriculum.STOP)]
    assert curriculum.factor == 1.0 and curriculum.finished


def test_single_level_curriculum_is_early_stopping() -> None:
    curriculum = _curriculum(levels=[1.0])
    steps = {epoch: curriculum.update(epoch, 10.0) for epoch in range(1, 10)}
    assert steps[5] == curriculum.STOP and not any(steps[epoch] for epoch in range(1, 5))


def test_curriculum_holds_while_the_reward_improves() -> None:
    curriculum = _curriculum()
    assert not any(curriculum.update(epoch, float(epoch) * 10.0) for epoch in range(1, 20))
    assert curriculum.factor == 0.0


def test_curriculum_needs_levels() -> None:
    with pytest.raises(ValueError, match="levels"):
        RewardPlateauCurriculum({"levels": []})


class _Module(BaseModule):
    pass


def test_module_manager_hands_the_factor_to_every_module() -> None:
    a, b = _Module(), _Module()
    manager = ModuleManager({"init_modules": [a], "pre_physics_step_modules": [b, a]})
    assert a.curriculum_factor == 1.0  # the full task by default (play, export)
    manager.set_curriculum_factor(0.25)
    assert a.curriculum_factor == b.curriculum_factor == manager.container.curriculum_factor == 0.25


def test_vlearn_env_forwards_the_agent_factor_when_it_changes() -> None:
    calls = []
    manager = SimpleNamespace(set_curriculum_factor=calls.append)
    env = VlearnEnv.__new__(VlearnEnv)
    env.envs = SimpleNamespace(env=SimpleNamespace(module_manager=manager))
    agent = SimpleNamespace(curriculum_factor=0.0)
    env.set_train_info(0, agent)
    env.set_train_info(100, agent)  # unchanged: not forwarded again
    agent.curriculum_factor = 0.5
    env.set_train_info(200, agent)
    assert calls == [0.0, 0.5]


def test_curriculum_ignores_the_warm_up_rewards() -> None:
    # A high reward in the first epochs (partial episodes) must not set the level's best and trigger a plateau.
    curriculum = _curriculum(min_epochs=5)
    rewards = [100.0, 50.0, 40.0, 45.0, 50.0] + [float(r) for r in range(60, 200, 10)]
    assert not any(curriculum.update(epoch, reward) for epoch, reward in enumerate(rewards, start=1))
    assert curriculum.factor == 0.0

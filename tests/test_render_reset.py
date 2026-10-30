"""Tests for the render menu's "Reset" checkbox and its trigger (render_reset_trigger) in the termination manager."""

from types import SimpleNamespace

import torch

from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.terminations.termination_manager_module import TerminationManagerModule
from optimal_morphology_rl.modules.visualization import render_module
from optimal_morphology_rl.modules.visualization.render_module import RenderModule


class _FakeCheckbox:
    def __init__(self, name: str, value: bool):
        self.name, self.value = name, value

    def get_value(self) -> bool:
        return self.value

    def set_value(self, value: bool) -> None:
        self.value = value


class _FakeGymRender:
    def __init__(self):
        self.items = []

    def register_menu_item(self, item):
        self.items.append(item)

    def reset_camera(self, eye, target):
        pass

    def set_paused(self, paused):
        pass

    def render(self, callback):
        return False


def _render(monkeypatch, **config):
    monkeypatch.setattr(render_module.v, "UserCheckbox", _FakeCheckbox)
    gym_render = _FakeGymRender()
    container = ModuleContainer()
    container.env = SimpleNamespace(rendering=True, gym=SimpleNamespace(get_render=lambda: gym_render))
    container.timestep, container.frame_skip = 0.005, 1
    module = RenderModule(config)
    module.post_finalize(container)
    return module, container, gym_render


def test_reset_checkbox_requests_a_reset_once(monkeypatch) -> None:
    module, container, gym_render = _render(monkeypatch)
    assert [item.name for item in gym_render.items] == ["Reset"]

    module.step(container)
    assert not container.get("render_reset_trigger")

    module.reset_checkbox.value = True
    module.step(container)
    assert container.render_reset_trigger is True
    assert module.reset_checkbox.value is False  # acts as a push button


def test_reset_button_can_be_disabled(monkeypatch) -> None:
    module, _, gym_render = _render(monkeypatch, reset_button=False)
    assert gym_render.items == [] and module.reset_checkbox is None


def test_termination_manager_truncates_every_env_on_a_render_reset() -> None:
    container = ModuleContainer()
    container.total_num_envs, container.device = 3, torch.device("cpu")
    module = TerminationManagerModule({})
    module.finalize(container)
    env = SimpleNamespace(term_buf=container.term_buf, trunc_buf=container.trunc_buf, info={})
    container.env = env

    module.step(container)
    assert not container.reset_buf.any()

    container.render_reset_trigger = True
    module.step(container)
    assert container.reset_buf.all() and env.trunc_buf.all()
    assert container.render_reset_trigger is False

    module.step(container)  # the request is consumed
    assert not container.reset_buf.any()


def test_render_reset_restarts_the_episode_timer_of_a_single_env() -> None:
    # Play and interactive mode run one env: its timer restarts from 0 on the Reset button.
    container = ModuleContainer()
    container.total_num_envs, container.device = 1, torch.device("cpu")
    module = TerminationManagerModule({"progress": {"max_episode_length": 10}})
    module.finalize(container)
    module.post_finalize(container)
    env = SimpleNamespace(term_buf=container.term_buf, trunc_buf=container.trunc_buf, progress_buf=container.progress_buf, info={})
    container.env = env

    for _ in range(4):
        module.step(container)
    container.render_reset_trigger = True
    module.step(container)  # the step on which Reset was pressed
    assert container.reset_buf.all()
    module.reset(container)  # the reset that follows
    assert container.progress_buf.tolist() == [0]

    for _ in range(9):
        module.step(container)
        assert not container.reset_buf.any()
    module.step(container)  # max_episode_length steps after the reset
    assert container.reset_buf.all()

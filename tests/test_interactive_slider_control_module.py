"""Unit tests for the interactive_slider_control module."""

import pytest
import torch

from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.visualization import interactive_slider_control_module
from optimal_morphology_rl.modules.visualization.interactive_slider_control_module import (
    InteractiveSliderControlModule,
)


class _FakeSlider:
    def __init__(self, name: str, slider_min: float, slider_max: float, value: float):
        self.name = name
        self.slider_min = slider_min
        self.slider_max = slider_max
        self.value = value

    def get_value(self) -> float:
        return self.value


class _FakeCheckbox:
    def __init__(self, name: str, value: bool):
        self.name = name
        self.value = value


class _FakeGymRender:
    def __init__(self):
        self.items: list = []

    def register_menu_item(self, item):
        self.items.append(item)


class _FakeEnv:
    def __init__(self, rendering: bool = True):
        self.rendering = rendering


@pytest.fixture
def container(monkeypatch) -> ModuleContainer:
    monkeypatch.setattr(interactive_slider_control_module.v, "UserSlider", _FakeSlider)
    monkeypatch.setattr(interactive_slider_control_module.v, "UserCheckbox", _FakeCheckbox)

    cont = ModuleContainer()
    cont.total_num_envs = 4
    cont.device = torch.device("cpu")
    cont.env = None
    cont.robot = object()
    cont.gym_render = _FakeGymRender()
    cont.num_actions = 3
    cont.actions = torch.zeros((4, 3), dtype=torch.float32)
    cont.reset_buf = torch.zeros(4, dtype=torch.bool)
    return cont


def _make_module(**overrides) -> InteractiveSliderControlModule:
    config = {"slider_min": 0.0, "slider_max": 10.0}
    config.update(overrides)
    return InteractiveSliderControlModule(config)


def test_finalize_requires_robot(container: ModuleContainer) -> None:
    container.robot = None
    module = _make_module()
    with pytest.raises(RuntimeError, match="requires 'robot'"):
        module.finalize(container)


def test_finalize_requires_slider_range_when_enabled(container: ModuleContainer) -> None:
    module = InteractiveSliderControlModule({})
    with pytest.raises(RuntimeError, match="slider_min"):
        module.finalize(container)


def test_disabled_module_allows_missing_slider_range(container: ModuleContainer) -> None:
    module = _make_module(enabled=False)
    module.finalize(container)


def test_post_finalize_noops_without_rendering(container: ModuleContainer) -> None:
    module = _make_module()
    module.finalize(container)
    module.post_finalize(container)

    assert module.sliders == []
    assert module.reset_checkbox is None
    assert container.gym_render.items == []


def test_post_finalize_noops_without_gym_render(container: ModuleContainer) -> None:
    container.env = _FakeEnv()
    container.gym_render = None  # type: ignore[assignment]
    module = _make_module()
    module.finalize(container)
    module.post_finalize(container)

    assert module.sliders == []
    assert module.reset_checkbox is None


def test_post_finalize_noops_without_num_actions(container: ModuleContainer) -> None:
    container.env = _FakeEnv()
    container.num_actions = None
    module = _make_module()
    module.finalize(container)
    module.post_finalize(container)

    assert module.sliders == []


def test_post_finalize_creates_generic_sliders(container: ModuleContainer) -> None:
    container.env = _FakeEnv()
    module = _make_module(slider_min=-1.0, slider_max=2.0)
    module.finalize(container)
    module.post_finalize(container)

    checkbox = container.gym_render.items[0]
    assert isinstance(checkbox, _FakeCheckbox)
    assert checkbox.name == "Reset"

    assert len(module.sliders) == 3
    for i, slider in enumerate(module.sliders):
        assert slider.name == f"Action {i}"
        assert slider.slider_min == -1.0
        assert slider.slider_max == 2.0
        assert slider.value == 0.0


def test_step_writes_slider_values_to_actions(container: ModuleContainer) -> None:
    container.env = _FakeEnv()
    module = _make_module()
    module.finalize(container)
    module.post_finalize(container)

    for value, slider in zip([1.0, 2.0, 3.0], module.sliders):
        slider.value = value

    module.step(container)

    assert container.actions.shape == (4, 3)
    assert torch.allclose(container.actions, torch.tensor([[1.0, 2.0, 3.0]]).expand(4, -1))

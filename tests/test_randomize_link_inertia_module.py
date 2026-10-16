"""Unit tests for the randomize_link_inertia module (link mass and diagonal inertia)."""

from types import SimpleNamespace

import pytest
import torch
import vlearn as v

from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.randomize_link_inertia_module import RandomizeLinkInertiaModule

LINKS = [("Palm", 0.4, (3.0, 4.0, 2.0)), ("MCP", 0.03, (2.0, 2.5, 2.5)), ("DP", 0.01, (1.0, 1.0, 1.0)), ("mcp_slider", 1e-6, (1e-6, 1e-6, 1e-6))]


class _Gym:
    def __init__(self):
        self.writes = []

    def create_gpu_array(self, commands):
        return commands[0]

    def set_link_properties(self, command):
        self.writes.append(command[0])


@pytest.fixture
def make(monkeypatch):
    # wrap_gpu_buffer asserts cuda tensors; tests run on cpu with fake commands.
    monkeypatch.setattr(v, "wrap_gpu_buffer", lambda buf: buf)

    def _make(config, num_envs=(1, 1, 2), with_spool=True):
        total = sum(num_envs)
        link_defs = [
            SimpleNamespace(name=name, mass=mass, diag_inertia=SimpleNamespace(x=i[0], y=i[1], z=i[2])) for name, mass, i in LINKS
        ]
        container = ModuleContainer()
        container.device = torch.device("cpu")
        container.num_envs = list(num_envs)
        container.total_num_envs = total
        container.robot = SimpleNamespace(
            num_links=len(LINKS), def_handle=3, art_def=SimpleNamespace(get_link_def=lambda i: link_defs[i]), link_masses=None
        )
        if with_spool:
            container.spool = SimpleNamespace(slider_link_names=["mcp_slider"])
        container.gym = _Gym()
        container.env_group = SimpleNamespace(
            create_link_property_command=lambda prop, buf, handle, index_range: (prop, buf, handle, index_range)
        )
        container.reset_buf = torch.ones(total, dtype=torch.bool)
        module = RandomizeLinkInertiaModule(config)
        module.finalize(container)
        module.post_finalize(container)
        return module, container

    return _make


CONFIG = {"mass_scale_range": [0.8, 1.2], "inertia_scale_range": [0.5, 2.0]}
DEFINED_MASS = torch.tensor([m for _, m, _ in LINKS])
DEFINED_INERTIA = torch.tensor([i for _, _, i in LINKS])


def test_defaults_select_every_link_but_the_spool_sliders(make) -> None:
    module, _ = make(CONFIG)
    assert module.selected == ["Palm", "MCP", "DP"]
    module, _ = make(CONFIG, with_spool=False)
    assert module.selected == ["Palm", "MCP", "DP", "mcp_slider"]


def test_reset_writes_mass_then_inertia_with_the_inertia_following_the_mass(make) -> None:
    module, container = make(CONFIG)
    module.reset(container)

    assert container.gym.writes == [v.LinkProperty.MASS, v.LinkProperty.DIAG_INERTIA]
    mass = container.set_link_mass_buf
    inertia = container.set_link_inertia_buf.view(3, len(LINKS), 3)
    mass_scale = mass / DEFINED_MASS
    inertia_scale = inertia / (DEFINED_INERTIA * mass_scale.unsqueeze(-1))
    selected = torch.tensor([True, True, True, False])
    assert torch.all((mass_scale[:, selected] >= 0.8) & (mass_scale[:, selected] <= 1.2))
    assert torch.all((inertia_scale[:, selected] >= 0.5 - 1e-6) & (inertia_scale[:, selected] <= 2.0 + 1e-6))
    # One inertia scale per link: the principal-axis ratios are kept.
    assert torch.allclose(inertia_scale[..., 0], inertia_scale[..., 2]) and torch.allclose(inertia_scale[..., 1], inertia_scale[..., 2])
    # The slider keeps its defined mass and inertia.
    assert torch.allclose(mass[:, 3], DEFINED_MASS[3].expand(3)) and torch.allclose(inertia[:, 3], DEFINED_INERTIA[3].expand(3, 3))
    # robot.link_masses is per env and follows the env sets (envs 2 and 3 share set 2).
    assert torch.equal(container.robot.link_masses, mass[[0, 1, 2, 2]])


def test_partial_reset_resamples_only_the_env_sets_of_resetting_envs(make) -> None:
    module, container = make(CONFIG)
    module.reset(container)
    before = container.set_link_mass_buf.clone()

    container.reset_buf[:] = torch.tensor([False, True, False, False])
    module.reset(container)
    after = container.set_link_mass_buf.clone()
    assert torch.equal(after[[0, 2]], before[[0, 2]])
    assert not torch.equal(after[1], before[1])

    container.reset_buf[:] = torch.tensor([False, False, False, True])  # env 3 is in set 2 with env 2
    module.reset(container)
    assert not torch.equal(container.set_link_mass_buf[2], after[2])
    assert torch.equal(container.robot.link_masses[2], container.robot.link_masses[3])


def test_no_reset_is_a_noop(make) -> None:
    module, container = make(CONFIG)
    container.reset_buf[:] = False
    module.reset(container)
    assert container.gym.writes == []
    assert torch.allclose(container.set_link_mass_buf, DEFINED_MASS.expand(3, -1))


def test_inertia_only_keeps_the_mass(make) -> None:
    module, container = make({"inertia_scale_range": [2.0, 2.0]})
    module.reset(container)
    assert torch.allclose(container.set_link_mass_buf, DEFINED_MASS.expand(3, -1))
    expected = DEFINED_INERTIA.clone()
    expected[:3] *= 2.0
    assert torch.allclose(container.set_link_inertia_buf.view(3, len(LINKS), 3), expected.expand(3, -1, -1))


def test_explicit_links(make) -> None:
    module, container = make({**CONFIG, "links": ["DP"]})
    module.reset(container)
    assert module.selected == ["DP"]
    assert torch.allclose(container.set_link_mass_buf[:, :2], DEFINED_MASS[:2].expand(3, -1))


@pytest.mark.parametrize(
    "links, match",
    [(["Thumb"], "not robot links"), (["mcp_slider"], "create_spool' sliders")],
)
def test_invalid_links_raise(make, links, match) -> None:
    with pytest.raises(RuntimeError, match=match):
        make({**CONFIG, "links": links})


@pytest.mark.parametrize("value", [[1.2, 0.8], [0.0, 1.0], 1.0])
def test_invalid_ranges_raise(value) -> None:
    with pytest.raises(RuntimeError, match="mass_scale_range"):
        RandomizeLinkInertiaModule({"mass_scale_range": value})

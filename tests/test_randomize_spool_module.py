"""Unit tests for the randomize_spool module (pulley radius and slider armature of the spool spools)."""

from types import SimpleNamespace

import pytest
import torch
import vlearn as v

from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.randomize_spool_module import RandomizeSpoolModule

NUM_DOFS = 5
SLIDER_DOFS = [3, 4]
DEFINED_ARMATURE = [0.1, 0.2, 0.3, 43.6, 43.6]


class _Spool:
    """The spool state 'randomize_spool' touches, with CreateSpoolModule's inertia update."""

    def __init__(self, num_envs: int):
        self.dof_indices = torch.tensor(SLIDER_DOFS)
        self.pully_radius = torch.full((num_envs, 2), 0.005)
        self.slider_armature = torch.full((num_envs, 2), 43.6)
        self.slider_mass = torch.tensor([1e-6, 1e-6])
        self.j_eff = torch.zeros((num_envs, 2))
        self.update_inertia()

    def update_inertia(self) -> None:
        self.j_eff[:] = (self.slider_armature + self.slider_mass) * self.pully_radius**2


class _Gym:
    def __init__(self):
        self.calls = 0

    def create_gpu_array(self, commands):
        return commands

    def set_joint_dof_properties(self, array):
        self.calls += 1


class _EnvGroup:
    def __init__(self):
        self.commands = []

    def create_joint_dof_property_command(self, prop, buf, handle, index_range):
        self.commands.append((prop, buf, handle, index_range))
        return (prop, buf)


@pytest.fixture
def make(monkeypatch):
    # wrap_gpu_buffer asserts cuda tensors; tests run on cpu with fake commands.
    monkeypatch.setattr(v, "wrap_gpu_buffer", lambda buf: buf)

    def _make(config, num_envs=(1, 1, 2)):
        total = sum(num_envs)
        spool = _Spool(total)
        container = ModuleContainer()
        container.device = torch.device("cpu")
        container.num_envs = list(num_envs)
        container.total_num_envs = total
        container.spool = spool
        container.robot = SimpleNamespace(
            num_joints=NUM_DOFS,
            def_handle=7,
            art_def=SimpleNamespace(get_joint_dof_def=lambda i: SimpleNamespace(armature=DEFINED_ARMATURE[i])),
        )
        container.gym = _Gym()
        container.env_group = _EnvGroup()
        container.reset_buf = torch.ones(total, dtype=torch.bool)
        return RandomizeSpoolModule(config), container, spool

    return _make


CONFIG = {"pully_radius_range": [0.0045, 0.0055], "armature_range": [35.0, 52.0]}


def test_full_reset_samples_both_and_updates_the_servo_inertia(make) -> None:
    module, container, spool = make(CONFIG)
    module.reset(container)

    assert torch.all((spool.pully_radius >= 0.0045) & (spool.pully_radius <= 0.0055))
    assert torch.all((spool.slider_armature >= 35.0) & (spool.slider_armature <= 52.0))
    prop, buf, handle, index_range = container.env_group.commands[0]
    assert prop == v.JointDofProperty.ARMATURE and handle == 7 and index_range == (0, NUM_DOFS)
    assert buf.shape == (3, NUM_DOFS)  # one row per env set
    assert torch.allclose(buf[:, :3], torch.tensor(DEFINED_ARMATURE[:3]).expand(3, 3))  # other DOFs keep their armature
    assert container.gym.calls == 1
    # Envs 2 and 3 share env set 2, so they share its armature.
    assert torch.equal(spool.slider_armature[2], spool.slider_armature[3])
    assert torch.equal(spool.slider_armature, buf[[0, 1, 2, 2]][:, SLIDER_DOFS])
    assert torch.allclose(spool.j_eff, (spool.slider_armature + spool.slider_mass) * spool.pully_radius**2)


def test_partial_reset_only_changes_resetting_envs_and_their_env_sets(make) -> None:
    module, container, spool = make(CONFIG)
    module.reset(container)
    radius, armature = spool.pully_radius.clone(), spool.slider_armature.clone()

    container.reset_buf[:] = torch.tensor([False, True, False, False])
    module.reset(container)

    assert torch.equal(spool.pully_radius[[0, 2, 3]], radius[[0, 2, 3]])
    assert not torch.equal(spool.pully_radius[1], radius[1])
    assert torch.equal(spool.slider_armature[[0, 2, 3]], armature[[0, 2, 3]])
    assert not torch.equal(spool.slider_armature[1], armature[1])


def test_reset_in_a_shared_env_set_resamples_the_whole_set(make) -> None:
    module, container, spool = make(CONFIG)
    module.reset(container)
    armature = spool.slider_armature.clone()

    container.reset_buf[:] = torch.tensor([False, False, True, False])
    module.reset(container)

    assert not torch.equal(spool.slider_armature[3], armature[3])  # env 3 shares env set 2 with env 2
    assert torch.equal(spool.slider_armature[2], spool.slider_armature[3])


def test_radius_only_does_not_touch_the_simulator(make) -> None:
    module, container, spool = make({"pully_radius_range": [0.004, 0.006]})
    module.reset(container)

    assert container.gym.calls == 0 and container.env_group.commands == []
    assert torch.all(spool.slider_armature == 43.6)
    assert torch.all((spool.pully_radius >= 0.004) & (spool.pully_radius <= 0.006))
    assert torch.allclose(spool.j_eff, (spool.slider_armature + spool.slider_mass) * spool.pully_radius**2)


def test_no_reset_is_a_noop(make) -> None:
    module, container, spool = make(CONFIG)
    container.reset_buf[:] = False
    radius = spool.pully_radius.clone()
    module.reset(container)
    assert torch.equal(spool.pully_radius, radius) and container.gym.calls == 0


@pytest.mark.parametrize(
    "config, match",
    [
        ({}, "pully_radius_range' and/or 'armature_range"),
        ({"pully_radius_range": [0.006, 0.004]}, "pully_radius_range"),
        ({"armature_range": [0.0, 10.0]}, "armature_range"),
        ({"armature_range": 40.0}, "armature_range"),
    ],
)
def test_invalid_config_raises(config, match) -> None:
    with pytest.raises(RuntimeError, match=match):
        RandomizeSpoolModule(config)


def test_missing_spool_raises(make) -> None:
    module, container, _ = make(CONFIG)
    container.spool = None
    with pytest.raises(RuntimeError, match="create_spool"):
        module.reset(container)


def test_scale_ranges_are_relative_to_each_tendons_nominal(make) -> None:
    module, container, spool = make({"pully_radius_scale_range": [0.8, 1.2], "armature_scale_range": [0.9, 1.1]})
    spool.pully_radius[:, 1] = 0.0025  # a smaller spool on tendon 1
    module.reset(container)
    assert torch.all((spool.pully_radius[:, 0] >= 0.004) & (spool.pully_radius[:, 0] <= 0.006))
    assert torch.all((spool.pully_radius[:, 1] >= 0.002) & (spool.pully_radius[:, 1] <= 0.003))
    assert torch.all((spool.slider_armature >= 0.9 * 43.6) & (spool.slider_armature <= 1.1 * 43.6))

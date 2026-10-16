"""Unit tests for the randomize_joint_friction module."""

import pytest
import torch
import vlearn as v

from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.randomize_joint_friction_module import RandomizeJointFrictionModule


class _FakeEnv:
    pass


class _FakeRobot:
    def __init__(self, num_joints: int = 3, fixed_hand: bool = True):
        self.use_tendon = False
        self.fixed_hand = fixed_hand
        self.num_motors = num_joints
        self.num_joints = num_joints
        self.def_handle = 42
        # Last DOF stands in for a non-revolute (e.g. tendon slider) DOF that keeps its defined friction.
        self.revolute_dof_mask = torch.ones(num_joints, dtype=torch.bool)
        self.revolute_dof_mask[-1] = num_joints == 1
        self.art_def = _FakeArtDef(num_joints)


class _FakeJointDef:
    def __init__(self, index: int):
        self.dof_start_index = index
        self.dof = 1
        self.joint_friction = 0.5


class _FakeArtDef:
    def __init__(self, num_joints: int):
        self.joint_defs = [_FakeJointDef(i) for i in range(num_joints)]

    def get_num_joint_defs(self) -> int:
        return len(self.joint_defs)

    def get_joint_def(self, i: int) -> _FakeJointDef:
        return self.joint_defs[i]


class _FakeGym:
    def __init__(self):
        self.calls: list[tuple[str, object]] = []

    def create_gpu_array(self, cmds):
        return ("gpu_array", cmds[0])

    def set_joint_properties(self, arr):
        self.calls.append(("set_joint_properties", arr))


class _FakeEnvGroup:
    def __init__(self):
        self.commands: list[tuple] = []

    def create_joint_property_command(self, prop, buf, handle, index_range, masks_buffer=None):
        self.commands.append((prop, buf, handle, index_range, masks_buffer))
        return ("joint_cmd", buf)


@pytest.fixture
def container(monkeypatch) -> ModuleContainer:
    # wrap_gpu_buffer asserts cuda tensors; tests run on cpu with fake commands.
    monkeypatch.setattr(v, "wrap_gpu_buffer", lambda buf: ("wrapped", buf))
    cont = ModuleContainer()
    cont.total_num_envs = 4
    cont.num_envs = [4]
    cont.device = torch.device("cpu")
    cont.env = _FakeEnv()
    cont.gym = _FakeGym()
    cont.env_group = _FakeEnvGroup()
    cont.robot = _FakeRobot()
    cont.reset_buf = torch.zeros(4, dtype=torch.bool)
    return cont


def _make_module(joint_friction_range) -> RandomizeJointFrictionModule:
    return RandomizeJointFrictionModule({"joint_friction_range": joint_friction_range})


def test_post_finalize_builds_command_and_samples(container: ModuleContainer) -> None:
    module = _make_module([0.05, 0.2])
    module.finalize(container)
    torch.manual_seed(0)
    module.post_finalize(container)

    assert container.gpu_set_joint_friction_cmd[0] == "gpu_array"
    prop, buf, handle, index_range, masks = container.env_group.commands[0]
    assert prop == v.JointProperty.JOINT_FRICTION
    assert handle == 42
    assert index_range == (0, 3)
    assert masks is None

    buf = container.set_joint_friction_buf
    assert buf.shape == (1, 3)
    assert torch.all(buf[:, :2] >= 0.05)
    assert torch.all(buf[:, :2] <= 0.2)
    assert torch.all(buf[:, 2] == 0.5)  # non-revolute DOF keeps its defined friction
    assert container.env_group.commands[0][1] == ("wrapped", container.set_joint_friction_buf)


def test_reset_resamples_and_applies(container: ModuleContainer) -> None:
    module = _make_module([0.05, 0.2])
    module.finalize(container)
    torch.manual_seed(0)
    module.post_finalize(container)

    container.reset_buf = torch.tensor([True, False, False, False])
    torch.manual_seed(1)
    module.reset(container)

    after = container.set_joint_friction_buf[:, :2]
    assert torch.all(after >= 0.05)
    assert torch.all(after <= 0.2)
    assert not torch.allclose(after, torch.full_like(after, 0.2))
    assert container.gym.calls[-1] == ("set_joint_properties", container.gpu_set_joint_friction_cmd)


def test_pinned_range_keeps_constant(container: ModuleContainer) -> None:
    module = _make_module([0.1, 0.1])
    module.finalize(container)
    torch.manual_seed(0)
    module.post_finalize(container)

    container.reset_buf = torch.ones(4, dtype=torch.bool)
    module.reset(container)

    assert torch.allclose(container.set_joint_friction_buf, torch.tensor([[0.1, 0.1, 0.5]]))
    assert container.gym.calls[-1][0] == "set_joint_properties"


def test_reset_is_noop_when_nothing_resets(container: ModuleContainer) -> None:
    module = _make_module([0.05, 0.2])
    module.finalize(container)
    module.post_finalize(container)

    module.reset(container)

    assert container.gym.calls == []


def test_finalize_requires_robot(container: ModuleContainer) -> None:
    container.robot = None
    module = _make_module([0.05, 0.2])
    with pytest.raises(RuntimeError, match="requires 'robot'"):
        module.finalize(container)


def test_post_finalize_requires_reset_buf(container: ModuleContainer) -> None:
    container.reset_buf = None
    module = _make_module([0.05, 0.2])
    module.finalize(container)
    with pytest.raises(RuntimeError, match="requires 'reset_buf'"):
        module.post_finalize(container)


def test_post_finalize_requires_range_config(container: ModuleContainer) -> None:
    module = RandomizeJointFrictionModule({})
    module.finalize(container)
    with pytest.raises(RuntimeError, match="joint_friction_range"):
        module.post_finalize(container)


def test_post_finalize_rejects_min_greater_than_max(container: ModuleContainer) -> None:
    module = _make_module([0.2, 0.05])
    module.finalize(container)
    with pytest.raises(RuntimeError, match="joint_friction_range"):
        module.post_finalize(container)


def test_post_finalize_rejects_malformed_range(container: ModuleContainer) -> None:
    module = _make_module(0.5)
    module.finalize(container)
    with pytest.raises(RuntimeError, match="joint_friction_range"):
        module.post_finalize(container)


def test_supports_robots_with_various_dof_counts(monkeypatch) -> None:
    # wrap_gpu_buffer asserts cuda tensors; tests run on cpu with fake commands.
    monkeypatch.setattr(v, "wrap_gpu_buffer", lambda buf: ("wrapped", buf))
    cont = ModuleContainer()
    cont.total_num_envs = 64
    cont.num_envs = [64]
    cont.device = torch.device("cpu")
    cont.env = _FakeEnv()
    cont.gym = _FakeGym()
    cont.env_group = _FakeEnvGroup()
    cont.robot = _FakeRobot(num_joints=6, fixed_hand=True)
    cont.reset_buf = torch.zeros(64, dtype=torch.bool)

    module = _make_module([0.05, 0.2])
    module.finalize(cont)
    module.post_finalize(cont)
    assert cont.env_group.commands[0][3] == (0, 6)
    assert cont.set_joint_friction_buf.shape == (1, 6)

    cont.robot = _FakeRobot(num_joints=2, fixed_hand=False)
    cont.reset_buf = torch.zeros(64, dtype=torch.bool)
    module = _make_module([0.05, 0.2])
    module.finalize(cont)
    module.post_finalize(cont)
    assert cont.env_group.commands[1][3] == (0, 2)
    assert cont.set_joint_friction_buf.shape == (1, 2)


def test_reset_resamples_only_env_sets_with_a_resetting_env(container: ModuleContainer) -> None:
    container.num_envs = [1, 1, 2]  # env 0 -> set 0, env 1 -> set 1, envs 2, 3 -> set 2
    module = _make_module([0.05, 0.2])
    module.finalize(container)
    module.post_finalize(container)
    before = container.set_joint_friction_buf.clone()
    assert before.shape == (3, 3)

    container.reset_buf = torch.tensor([False, True, False, True])
    module.reset(container)

    after = container.set_joint_friction_buf
    assert torch.equal(after[0], before[0])
    assert not torch.equal(after[1, :2], before[1, :2])
    assert not torch.equal(after[2, :2], before[2, :2])
    assert torch.all(after[:, 2] == 0.5)
    assert container.gym.calls[-1] == ("set_joint_properties", container.gpu_set_joint_friction_cmd)

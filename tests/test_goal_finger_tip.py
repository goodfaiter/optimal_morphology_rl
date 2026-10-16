"""Unit tests for the goal_finger_tip observation, reward, and sensor binding."""

import math

import pytest
import torch

import optimal_morphology_rl.modules.create_kinematic_sensor_module as kinematic_module
from optimal_morphology_rl.modules.create_kinematic_sensor_module import CreateKinematicSensorModule
from optimal_morphology_rl.modules.observations.goal_finger_tip_observation import (
    GoalFingerTipObservation,
)
from optimal_morphology_rl.modules.rewards.goal_finger_tip_reward import GoalFingerTipReward


class _FakeContainer:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def get(self, key: str, default=None):
        return getattr(self, key, default)


class _FakeModuleManager:
    def __init__(self, container):
        self.container = container


class _FakeEnv:
    def __init__(self, container, total_num_envs: int = 1, device: str = "cpu"):
        self.module_manager = _FakeModuleManager(container)
        self.total_num_envs = total_num_envs
        self.device = torch.device(device)
        self.info = {"rewards": {}}


# ---------------------------------------------------------------------------
# goal_finger_tip observation
# ---------------------------------------------------------------------------
def test_get_obs_dim() -> None:
    container = _FakeContainer()
    env = _FakeEnv(container)
    module = GoalFingerTipObservation({})
    assert module.get_obs_dim(env) == 3


def test_obs_raises_when_goal_pose_buffer_missing() -> None:
    container = _FakeContainer()
    env = _FakeEnv(container)
    module = GoalFingerTipObservation({})

    out = torch.zeros((1, 3))
    with pytest.raises(RuntimeError, match="goal_pose"):
        module.compute_observation(env, out)


def test_observes_goal_from_container() -> None:
    goal = torch.tensor([[0.05, -0.12, 0.25]])
    container = _FakeContainer(goal_position_in_goal_frame=goal)
    env = _FakeEnv(container)
    module = GoalFingerTipObservation({})

    out = torch.zeros((1, 3))
    module.compute_observation(env, out)

    assert torch.allclose(out, goal)
    # The observation must not own or mutate the goal buffer.
    module.compute_observation(env, out)
    assert torch.allclose(out, goal)


# ---------------------------------------------------------------------------
# goal_finger_tip reward
# ---------------------------------------------------------------------------
class _FakeKinematicSensor:
    def __init__(self, pos: torch.Tensor):
        self.pos_in_world = pos


def _reward_container(fingertip_pos: torch.Tensor, goal_pos: torch.Tensor) -> tuple[_FakeContainer, _FakeEnv]:
    container = _FakeContainer()
    container.kinematic_sensors = {"Pinky_DP": _FakeKinematicSensor(fingertip_pos)}
    # Goal frame = environment frame: the reward converts the goal with goal_in_env_frame.
    container.goal_position_in_goal_frame = goal_pos
    container.goal_orientation_in_goal_frame = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(goal_pos.shape[0], 4)
    container.goal_frame_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(goal_pos.shape[0], 4)
    container.goal_frame_pos = torch.zeros_like(goal_pos)
    env = _FakeEnv(container)
    return container, env


def test_observation_uses_the_goal_frame_and_reward_the_env_frame() -> None:
    s = math.sqrt(0.5)  # goal frame rotated 90 deg about z and shifted: local x -> env y
    goal_local = torch.tensor([[0.5, 0.0, 0.0]])
    container = _FakeContainer(
        goal_position_in_goal_frame=goal_local,
        goal_orientation_in_goal_frame=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        goal_frame_quat=torch.tensor([[0.0, 0.0, s, s]]),
        goal_frame_pos=torch.tensor([[1.0, 2.0, 3.0]]),
        kinematic_sensors={"Pinky_DP": _FakeKinematicSensor(torch.tensor([[1.0, 2.5, 3.0]]))},  # tip at the goal
    )
    env = _FakeEnv(container)
    out = torch.zeros(1, 3)
    GoalFingerTipObservation({}).compute_observation(env, out)
    assert torch.equal(out, goal_local)  # the observation is the goal-frame value, untouched

    reward = GoalFingerTipReward({"scale": 1.0, "normalize": 0.02, "link_name": "DP"}).compute(env)
    assert reward.item() == pytest.approx(1.0, abs=1e-5)  # the reward compared the tip with the env-frame goal


def test_reward_is_scale_at_exact_goal() -> None:
    _, env = _reward_container(
        torch.tensor([[0.05, -0.15, 0.25]]),
        torch.tensor([[0.05, -0.15, 0.25]]),
    )
    module = GoalFingerTipReward({"scale": 1.0, "normalize": 0.05, "link_name": "DP"})

    reward = module.compute(env)

    assert reward.item() == pytest.approx(1.0)
    assert env.info["rewards"]["goal_finger_tip_reward"] == pytest.approx(1.0)
    assert env.info["rewards"]["goal_finger_tip_error_l2_norm_mm"] == pytest.approx(0.0)


def test_reward_decays_with_distance() -> None:
    _, env = _reward_container(
        torch.tensor([[0.05, -0.15, 0.25]]),
        torch.tensor([[0.05, -0.15, 0.15]]),  # dist 0.1 m
    )
    module = GoalFingerTipReward({"scale": 2.0, "normalize": 0.05, "link_name": "DP"})

    reward = module.compute(env)

    expected = 2.0 * math.exp(-((0.1 / 0.05) ** 2))
    assert reward.item() == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# create_kinematic_sensor articulation binding
# ---------------------------------------------------------------------------
class _FakeLinkDef:
    def __init__(self, name: str):
        self.name = name


class _FakeArtDef:
    def __init__(self, link_names, kinematic_sensor_link_names):
        self._link_names = link_names
        self._kin_names = kinematic_sensor_link_names

    def get_num_kinematic_sensor_defs(self) -> int:
        return len(self._kin_names)

    def get_kinematic_sensor_def(self, i: int) -> object:
        return type("Def", (), {"link_name": self._kin_names[i]})()

    def get_num_link_defs(self) -> int:
        return len(self._link_names)

    def get_link_def(self, i: int) -> _FakeLinkDef:
        return _FakeLinkDef(self._link_names[i])

    def get_link_def_names(self) -> list[str]:
        return list(self._link_names)


class _FakeSource:
    def get_kinematic_sensor_handle(self, index: int) -> object:
        return ("handle", index)


class _FakeEnvDef:
    def get_rigid_body(self, handle):
        raise ValueError("not a rigid body")

    def get_articulation(self, handle):
        return _FakeSource()


class _FakeEnvGroup:
    def create_kinematic_sensor_state_command(self, transforms, velocities, handle, frame_type=None):
        return ("cmd", handle)


class _FakeGym:
    def create_gpu_array(self, cmds):
        return ("array", cmds)


def _sensor_container(kinematic_sensor_link_names, missing_reward_object: bool = True, monkeypatch=None) -> _FakeContainer:
    if monkeypatch is not None:
        # wrap_gpu_buffer asserts cuda tensors; tests run on cpu with fake commands.
        monkeypatch.setattr(kinematic_module.v, "wrap_gpu_buffer", lambda buf: ("wrapped", buf))
    container = _FakeContainer()
    container.env_def = _FakeEnvDef()
    container.env_group = _FakeEnvGroup()
    container.gym = _FakeGym()
    container.total_num_envs = 1
    container.device = torch.device("cpu")
    container.reward_object_name = "cube"

    class _FakeRobot:
        def __init__(self):
            self.arti_handle = 99
            self.art_def = _FakeArtDef(["palm", "distal"], kinematic_sensor_link_names)

    container.robot = _FakeRobot()
    if not missing_reward_object:

        class _FakeRewardObject:
            handle = 123
            name = "cube"

        container.reward_object = _FakeRewardObject()
    else:
        container.reward_object = None
    return container


def test_creates_all_articulation_sensors_keyed_by_link_name(monkeypatch) -> None:
    container = _sensor_container(["Pinky_DP", "Index_DP"], monkeypatch=monkeypatch)
    module = CreateKinematicSensorModule({})

    module.post_finalize(container)

    assert set(container.kinematic_sensors.keys()) == {"Pinky_DP", "Index_DP"}
    assert container.kinematic_sensors["Pinky_DP"].name == "Pinky_DP"
    assert "kinematic_sensor" not in container.__dict__  # no singular without a reward object


def test_find_kinematic_sensor_by_substring(monkeypatch) -> None:
    container = _sensor_container(["Pinky_DP", "Index_DP"], monkeypatch=monkeypatch)
    module = CreateKinematicSensorModule({})
    module.post_finalize(container)

    assert container.kinematic_sensors["Pinky_DP"].kinematic_sensor_handle == ("handle", 0)
    assert container.kinematic_sensors["Index_DP"].kinematic_sensor_handle == ("handle", 1)


def test_find_kinematic_sensor_substring_lookup(monkeypatch) -> None:
    container = _sensor_container(["Pinky_DP", "Index_DP"], monkeypatch=monkeypatch)
    module = CreateKinematicSensorModule({})
    module.post_finalize(container)

    from optimal_morphology_rl.modules.create_kinematic_sensor_module import find_kinematic_sensor

    assert find_kinematic_sensor(container, "DP").name == "Pinky_DP"
    assert find_kinematic_sensor(container, "index").name == "Index_DP"
    assert find_kinematic_sensor(container, "thumb") is None


def test_inert_when_articulation_has_no_sensors(monkeypatch) -> None:
    container = _sensor_container([], monkeypatch=monkeypatch)
    module = CreateKinematicSensorModule({})

    module.post_finalize(container)

    assert container.kinematic_sensors == {}


def test_reward_object_sensor_keyed_by_object_name(monkeypatch) -> None:
    container = _sensor_container([], missing_reward_object=False, monkeypatch=monkeypatch)
    module = CreateKinematicSensorModule({})

    module.post_finalize(container)

    # Keyed by the object's body name; no singular attribute anymore.
    assert set(container.kinematic_sensors.keys()) == {"cube"}
    assert "kinematic_sensor" not in container.__dict__


def test_module_post_finalize_kin_sensors_empty_without_reward_object(monkeypatch) -> None:
    container = _sensor_container(["Pinky_DP"], monkeypatch=monkeypatch)
    module = CreateKinematicSensorModule({})
    module.post_finalize(container)

    assert set(container.kinematic_sensors.keys()) == {"Pinky_DP"}


def test_reward_raises_when_fingertip_sensor_missing() -> None:
    from optimal_morphology_rl.modules.rewards.goal_finger_tip_reward import GoalFingerTipReward

    container = _FakeContainer()
    env = _FakeEnv(container)
    module = GoalFingerTipReward({"link_name": "DP", "scale": 1.0, "normalize": 0.05})

    with pytest.raises(RuntimeError, match="no kinematic sensor matches"):
        module.compute(env)


def test_vsim_emits_fingertip_kinematic_sensor() -> None:
    """The emission end-to-end: urdf_to_vsim writes the kinematicSensor block."""
    from optimal_morphology_rl_assets import urdf_to_vsim as u
    from yourdfpy import URDF

    urdf = URDF.load(str(u.INPUT_URDF), build_scene_graph=False, load_meshes=False)
    vsim = u.URDFVsimQueryWrapper(urdf)
    u.add_sensors(vsim)
    u.add_kinematic_sensors(vsim)

    text = u.urdf_to_text_rounded(vsim)
    assert "<kinematicSensor>" in text
    assert 'name="Pinky_DP_kine_sensor"' in text
    assert 'link="Pinky_DP"' in text
    assert 'flags="transform velocity"' in text

"""Unit tests for the goal_pose module (even split over discrete / sinusoidal / square goal types)."""

import math
from collections import Counter
from pathlib import Path

import pytest
import torch

from optimal_morphology_rl.modules import goal_pose_module
from optimal_morphology_rl.modules.goal_pose_module import GoalPoseModule, _rpy_to_quat
from optimal_morphology_rl.modules.module_container import ModuleContainer

RANGES = {"x": [0.0, 0.1], "y": [0.2, 0.2], "z": [0.3, 0.34]}


BLOCKS = {
    "discrete": {},
    "sinusoidal": {"sinusoidal": {"min_frequency": 0.2, "max_frequency": 0.8, "min_amplitude_fraction": 0.5}},
    "square": {"plane": [0, 2], "speed": {"min": 0.01, "max": 0.05}},
}


def _config(types, **overrides) -> dict:
    """Shared keys plus one ``types`` block per type in ``types`` (in order; repeats become the weight).

    Overrides named after a type are merged into its block; the others are shared keys.
    """
    blocks = {name: {**BLOCKS[name], **({"weight": count} if count > 1 else {})} for name, count in Counter(types).items()}
    config = {"goal_position": RANGES, "goal_orientation": [0.0, 0.0, 0.0, 1.0], "types": blocks}
    for key, value in overrides.items():
        if key in BLOCKS:
            blocks[key] = {**blocks.get(key, {}), **value}
        else:
            config[key] = value
    return config


def _container(num_envs: int = 6, timestep: float | None = 0.005) -> ModuleContainer:
    container = ModuleContainer()
    container.total_num_envs = num_envs
    container.device = torch.device("cpu")
    if timestep is not None:
        container.timestep = timestep
        container.frame_skip = 2
    container.reset_buf = torch.ones(num_envs, dtype=torch.bool)
    return container


def _module(types, num_envs: int = 6, **overrides) -> tuple[GoalPoseModule, ModuleContainer]:
    module = GoalPoseModule(_config(types, **overrides))
    container = _container(num_envs)
    module.post_finalize(container)
    module.reset(container)
    container.reset_buf[:] = False
    return module, container


def _in_ranges(pos: torch.Tensor) -> bool:
    lo = torch.tensor([RANGES[a][0] for a in "xyz"]) - 1e-6
    hi = torch.tensor([RANGES[a][1] for a in "xyz"]) + 1e-6
    return bool(torch.all((pos >= lo) & (pos <= hi)))


# ---------------------------------------------------------------- split


def test_envs_split_evenly_in_contiguous_blocks() -> None:
    module, _ = _module(["square", "discrete", "sinusoidal"], num_envs=7)
    sizes = {name: int(mask.sum()) for name, mask in module._masks.items()}
    assert sizes == {"square": 3, "discrete": 2, "sinusoidal": 2}
    assert module._masks["square"][:3].all() and module._masks["discrete"][3:5].all()
    total = sum(mask.long() for mask in module._masks.values())
    assert torch.all(total == 1)


def test_weight_gives_a_type_a_larger_share() -> None:
    module, _ = _module(["square", "square", "discrete"], num_envs=6)
    assert module.entries == ["square", "square", "discrete"]
    assert module._masks["square"].tolist() == [True, True, True, True, False, False]
    assert module._masks["discrete"].tolist() == [False, False, False, False, True, True]


def test_empty_type_block_uses_shared_keys() -> None:
    module = GoalPoseModule({"goal_position": RANGES, "types": {"discrete": None}})
    assert module.entries == ["discrete"]
    assert module.goal_types["discrete"].x_range == (0.0, 0.1)


def test_single_type_covers_all_envs() -> None:
    module, container = _module(["discrete"], num_envs=4)
    assert module._masks["discrete"].all()
    assert container.goal_position_in_goal_frame.shape == (4, 3) and container.goal_orientation_in_goal_frame.shape == (4, 4)


@pytest.mark.parametrize("types", [None, {}, ["discrete"]])
def test_config_without_types_mapping_raises(types) -> None:
    with pytest.raises(ValueError, match="'types' mapping"):
        GoalPoseModule({"goal_position": RANGES, "types": types})


def test_unknown_goal_type_raises() -> None:
    with pytest.raises(ValueError, match="unknown goal types"):
        GoalPoseModule({"goal_position": RANGES, "types": {"discrete": {}, "circle": {}}})


@pytest.mark.parametrize("weight", [0, -1, 1.5, "2"])
def test_invalid_weight_raises(weight) -> None:
    with pytest.raises(ValueError, match="weight"):
        GoalPoseModule(_config(["discrete"], discrete={"weight": weight}))


def test_shared_keys_are_overridden_by_type_blocks() -> None:
    module = GoalPoseModule(_config(["discrete", "square"], discrete={"goal_orientation": "random", "resample_prob": 0.5}))
    assert module.goal_types["discrete"].goal_orientation == "random"
    assert module.goal_types["discrete"].resample_prob == 0.5
    assert module.goal_types["square"].goal_orientation == [0.0, 0.0, 0.0, 1.0]


# ---------------------------------------------------------------- per-type behaviour


def test_reset_samples_every_type_within_ranges() -> None:
    _, container = _module(["square", "discrete", "sinusoidal"], num_envs=9)
    assert _in_ranges(container.goal_position_in_goal_frame)
    assert torch.allclose(container.goal_orientation_in_goal_frame, torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(9, 4))


def test_step_moves_only_the_moving_types() -> None:
    module, container = _module(["discrete", "sinusoidal", "square"], num_envs=6)
    before = container.goal_position_in_goal_frame.clone()
    for _ in range(20):
        module.step(container)
    after = container.goal_position_in_goal_frame
    discrete, moving = module._masks["discrete"], ~module._masks["discrete"]
    assert torch.equal(after[discrete], before[discrete])  # resample_prob defaults to 0
    assert torch.all((after[moving] - before[moving]).abs().sum(-1) > 0)
    assert _in_ranges(after)


def test_discrete_resample_prob_touches_only_discrete_envs() -> None:
    module, container = _module(["discrete", "square"], num_envs=6, discrete={"resample_prob": 1.0})
    square = module._masks["square"]
    module.goal_types["square"].step = lambda container, mask: None  # freeze square for the comparison
    before = container.goal_position_in_goal_frame.clone()
    torch.manual_seed(1)
    module.step(container)
    assert torch.equal(container.goal_position_in_goal_frame[square], before[square])
    assert not torch.equal(container.goal_position_in_goal_frame[~square], before[~square])


def test_square_stays_on_the_perimeter() -> None:
    module, container = _module(["square"], num_envs=8)
    for _ in range(200):
        module.step(container)
        pos = container.goal_position_in_goal_frame
        on_x_edge = torch.isclose(pos[:, 0], torch.tensor(0.0), atol=1e-6) | torch.isclose(pos[:, 0], torch.tensor(0.1), atol=1e-6)
        on_z_edge = torch.isclose(pos[:, 2], torch.tensor(0.3), atol=1e-6) | torch.isclose(pos[:, 2], torch.tensor(0.34), atol=1e-6)
        assert torch.all(on_x_edge | on_z_edge)
        assert torch.allclose(pos[:, 1], torch.tensor(0.2))


def test_sinusoid_follows_its_pattern() -> None:
    module, container = _module(["sinusoidal"], num_envs=4)
    goal = module.goal_types["sinusoidal"]
    for _ in range(10):
        module.step(container)
    t = torch.full((4, 1), 10 * 0.01)
    expected = goal._center + goal._amp * torch.sin(2 * math.pi * goal._freq * t + goal._phase)
    assert torch.allclose(container.goal_position_in_goal_frame, expected, atol=1e-6)
    assert torch.all(goal._amp[:, 1] == 0.0)  # degenerate y range


def test_partial_reset_only_changes_reset_envs_and_restarts_their_time() -> None:
    module, container = _module(["sinusoidal", "square"], num_envs=4)
    for _ in range(5):
        module.step(container)
    before = container.goal_position_in_goal_frame.clone()
    container.reset_buf[:] = torch.tensor([False, True, False, True])
    module.reset(container)
    assert torch.equal(container.goal_position_in_goal_frame[[0, 2]], before[[0, 2]])
    assert module.goal_types["sinusoidal"]._time_buf[1] == 0.0
    assert module.goal_types["square"]._time_buf[3] == 0.0
    assert module.goal_types["sinusoidal"]._time_buf[0] > 0.0


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"sinusoidal": {"sinusoidal": {"min_frequency": 0.0}}}, "min_frequency"),
        ({"sinusoidal": {"sinusoidal": {"min_amplitude_fraction": 0.9, "max_amplitude_fraction": 0.5}}}, "amplitude"),
        ({"square": {"speed": {"min": 0.0, "max": 0.05}}}, "min_speed"),
        ({"square": {"plane": [1, 1]}}, "plane"),
    ],
)
def test_invalid_type_config_raises(overrides, match) -> None:
    with pytest.raises(ValueError, match=match):
        GoalPoseModule(_config(["sinusoidal", "square"], **overrides))


def test_missing_timestep_raises_at_allocation() -> None:
    container = _container(2, timestep=None)
    with pytest.raises(RuntimeError, match="timestep"):
        GoalPoseModule(_config(["square"])).post_finalize(container)


# ---------------------------------------------------------------- goal frame


def test_goal_is_kept_in_the_goal_frame_and_converted_on_demand() -> None:
    s = math.sqrt(0.5)  # 90 deg about z: local x -> environment y, local y -> environment -x
    module, container = _module(["square", "sinusoidal"], num_envs=4)
    container.goal_frame_quat[:] = torch.tensor([0.0, 0.0, s, s])
    container.goal_frame_pos[:] = torch.tensor([1.0, 2.0, 3.0])
    for _ in range(3):
        module.step(container)

    local = container.goal_position_in_goal_frame
    assert _in_ranges(local)  # sampled within the configured ranges, no transform applied
    pos, quat = goal_pose_module.goal_in_env_frame(container)
    expected = torch.stack([1.0 - local[:, 1], 2.0 + local[:, 0], 3.0 + local[:, 2]], dim=-1)
    assert torch.allclose(pos, expected, atol=1e-6)
    # The goal orientation ([0, 0, 0, 1] in the goal frame) is the frame's orientation in the env frame.
    assert torch.allclose(quat, torch.tensor([[0.0, 0.0, s, s]] * 4), atol=1e-6)


VSIM = """<robot name="hand">
    <link name="Palm"/>
    <link name="Abd_jointbody"/>
    <link name="Abd"/>
    <link name="MCP_jointbody"/>
    <joint name="Abd" type="fixed">
        <parent link="Palm"/>
        <child link="Abd_jointbody"/>
        <origin xyz="0 1 0" rpy="0 0 1.5707963267948966"/>
    </joint>
    <joint name="Abd_offset" type="fixed">
        <parent link="Abd_jointbody"/>
        <child link="Abd"/>
        <origin xyz="1 0 0"/>
    </joint>
    <joint name="MCP" type="revolute">
        <parent link="Abd"/>
        <child link="MCP_jointbody"/>
        <origin xyz="0 0 1" rpy="0 0 0"/>
    </joint>
</robot>
"""


def test_frame_chain_walks_up_to_the_nearest_simulated_link(tmp_path: Path) -> None:
    vsim = tmp_path / "hand.vsim"
    vsim.write_text(VSIM)
    sim_links = ["Palm", "MCP_jointbody"]  # vlearn merges the fixed-joint links into Palm

    link, chain = goal_pose_module._frame_chain(str(vsim), "Abd", sim_links)
    assert link == "Palm"
    assert chain == [{"xyz": [0.0, 1.0, 0.0], "rpy": [0.0, 0.0, math.pi / 2]}, {"xyz": [1.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]}]
    assert goal_pose_module._frame_chain(str(vsim), "MCP_jointbody", sim_links) == ("MCP_jointbody", [])
    with pytest.raises(RuntimeError, match="is not a link"):
        goal_pose_module._frame_chain(str(vsim), "Thumb", sim_links)
    with pytest.raises(RuntimeError, match="no simulated ancestor"):
        goal_pose_module._frame_chain(str(vsim), "Abd", ["MCP_jointbody"])


def test_frame_pose_composes_the_simulated_link_and_the_vsim_chain(tmp_path: Path, monkeypatch) -> None:
    vsim = tmp_path / "hand.vsim"
    vsim.write_text(VSIM)

    class _Robot:
        num_links = 2
        arti_handle = None

        class art_def:
            @staticmethod
            def get_link_def(i):
                return type("LinkDef", (), {"name": ["Palm", "MCP_jointbody"][i]})

    class _Gym:
        def compute_kinematics(self):
            pass

        def create_gpu_array(self, commands):
            return commands

        def get_link_transforms(self, commands):
            commands[0][:] = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])  # Palm at x = 1, unrotated

    container = _container(2)
    container.robot = _Robot()
    container.robot_vsim_path = str(vsim)
    container.gym = _Gym()
    container.env_group = type("EnvGroup", (), {"create_link_transform_command": staticmethod(lambda buf, handle, rng: buf)})

    # wrap_gpu_buffer asserts cuda tensors; the fake gym works on the cpu tensor directly.
    monkeypatch.setattr(goal_pose_module.v, "wrap_gpu_buffer", lambda buf: buf)
    module = GoalPoseModule(_config(["discrete"], frame="Abd"))
    module.post_finalize(container)
    module.reset(container)

    # Palm (1, 0, 0) + Abd joint (0, 1, 0), then 1 m along the rotated x (= world y) -> (1, 2, 0).
    assert torch.allclose(container.goal_frame_pos, torch.tensor([[1.0, 2.0, 0.0]] * 2), atol=1e-6)
    assert torch.allclose(container.goal_frame_quat, torch.tensor([_rpy_to_quat([0.0, 0.0, math.pi / 2])] * 2), atol=1e-6)


def test_goal_frame_is_published_for_the_envs_of_framed_types(tmp_path: Path, monkeypatch) -> None:
    vsim = tmp_path / "hand.vsim"
    vsim.write_text(VSIM)

    class _Robot:
        num_links = 2
        arti_handle = None

        class art_def:
            @staticmethod
            def get_link_def(i):
                return type("LinkDef", (), {"name": ["Palm", "MCP_jointbody"][i]})

    class _Gym:
        def compute_kinematics(self):
            pass

        def create_gpu_array(self, commands):
            return commands

        def get_link_transforms(self, commands):
            commands[0][:] = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])

    container = _container(4)
    container.robot = _Robot()
    container.robot_vsim_path = str(vsim)
    container.gym = _Gym()
    container.env_group = type("EnvGroup", (), {"create_link_transform_command": staticmethod(lambda buf, handle, rng: buf)})
    monkeypatch.setattr(goal_pose_module.v, "wrap_gpu_buffer", lambda buf: buf)
    config = _config(["discrete", "square"])
    config["types"]["discrete"]["frame"] = "Abd"  # only the discrete envs (0, 1) use a frame
    module = GoalPoseModule(config)
    module.post_finalize(container)
    module.reset(container)

    assert torch.allclose(container.goal_frame_pos[:2], torch.tensor([[1.0, 2.0, 0.0]] * 2), atol=1e-6)
    assert torch.equal(container.goal_frame_pos[2:], torch.zeros(2, 3))
    assert torch.equal(container.goal_frame_quat[2:], torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2))

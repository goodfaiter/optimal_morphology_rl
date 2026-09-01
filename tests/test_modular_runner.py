"""Tests for the modular task runner without requiring a live simulation."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from optimal_morphology_rl.modules.articulation_link_colorer_module import (
    ArticulationLinkColorerModule,
)
from optimal_morphology_rl.modules.visualization.goal_visualization_module import (
    GoalVisualizationModule,
)
from optimal_morphology_rl.modules.visualization.render_module import RenderModule
from optimal_morphology_rl.runners.modular_runner import (
    adjust_minibatch_size,
    apply_env_overrides,
    apply_ppo_overrides,
    load_task_configs,
    sync_num_envs,
)


TASKS = [
    "hand_cube",
    "hand_drawer",
    "hand_button",
    "hand_button_difficult",
    "hand_tomato",
    "hand_tomato_extreme",
]


@pytest.fixture
def task_root() -> Path:
    return Path(__file__).resolve().parent.parent / "src" / "optimal_morphology_rl" / "envs"


@pytest.mark.parametrize("task", TASKS)
def test_task_configs_exist(task: str, task_root: Path) -> None:
    env_path, env_config, ppo_config = load_task_configs(
        task, task_root, context={"mode": "train"}
    )
    assert env_path.exists()
    assert env_config.get("modules")
    assert "create_rigid_vsim_envs" in env_config
    assert ppo_config.get("params", {}).get("config", {}).get("env_name") == f"{task}_env"


@pytest.mark.parametrize("task", TASKS)
def test_ppo_name_matches_task(task: str, task_root: Path) -> None:
    _, _, ppo_config = load_task_configs(task, task_root, context={"mode": "train"})
    cfg = ppo_config["params"]["config"]
    assert cfg["name"] == task
    assert cfg["env_name"] == f"{task}_env"


def test_apply_env_overrides_num_envs() -> None:
    env_config = {"create_rigid_vsim_envs": {"num_envs": 4096}}
    args = argparse.Namespace(
        num_envs=128,
        device="cuda:1",
        seed=42,
        mode="train",
        headless="True",
    )
    out = apply_env_overrides(env_config, args)
    sim = out["create_rigid_vsim_envs"]
    assert sim["num_envs"] == 128
    assert sim["device"] == "cuda:1"
    assert sim["seed"] == 42
    assert sim["rendering"] is False
    assert sim["with_window"] is False


def test_apply_ppo_overrides() -> None:
    ppo_config = {"params": {"config": {"num_actors": 4096}}}
    args = argparse.Namespace(
        seed=7,
        num_envs=256,
        max_epochs=10,
        horizon_length=64,
        learning_rate=1e-3,
        kl_threshold=0.01,
        experiment_name="test_exp",
        mode="train",
        games_num=None,
        deterministic=None,
    )
    out = apply_ppo_overrides(ppo_config, args, "hand_cube")
    cfg = out["params"]["config"]
    assert cfg["num_actors"] == 256
    assert cfg["max_epochs"] == 10
    assert cfg["horizon_length"] == 64
    assert cfg["learning_rate"] == 1e-3
    assert cfg["kl_threshold"] == 0.01
    assert cfg["full_experiment_name"] == "test_exp"
    assert cfg["player"]["use_vecenv"] is True


def test_apply_ppo_overrides_play_mode() -> None:
    ppo_config = {"params": {"config": {"num_actors": 4096}}}
    args = argparse.Namespace(
        seed=None,
        num_envs=None,
        max_epochs=None,
        horizon_length=None,
        learning_rate=None,
        kl_threshold=None,
        experiment_name=None,
        mode="play",
        games_num=5,
        deterministic="True",
    )
    out = apply_ppo_overrides(ppo_config, args, "hand_cube")
    cfg = out["params"]["config"]
    assert cfg["num_actors"] == 4
    assert cfg["player"]["games_num"] == 5
    assert cfg["player"]["deterministic"] is True


def test_sync_num_envs() -> None:
    env_config = {"create_rigid_vsim_envs": {"num_envs": 4096}}
    ppo_config = {"params": {"config": {"num_actors": 512}}}
    out = sync_num_envs(env_config, ppo_config)
    assert out["create_rigid_vsim_envs"]["num_envs"] == 512


def test_adjust_minibatch_size() -> None:
    cfg = {"minibatch_size": 4096}
    adjust_minibatch_size(cfg, num_envs=256, horizon_len=128)
    # batch = 32768; 8 batches of the original 4096 size fit exactly
    assert cfg["minibatch_size"] == 4096


def test_adjust_minibatch_size_division() -> None:
    cfg = {"minibatch_size": 4096}
    adjust_minibatch_size(cfg, num_envs=64, horizon_len=128)
    # batch = 8192, two batches of 4096
    assert cfg["minibatch_size"] == 4096


def test_render_module_defaults() -> None:
    module = RenderModule({})
    assert module.render_substep is True
    assert module.capped_step is False
    assert module.paused is False
    assert module.raise_exception is None
    assert module.camera["eye"] == [-0.671139, 0.073098, 0.726423]


def test_goal_visualization_module_defaults() -> None:
    module = GoalVisualizationModule({})
    assert module.line_width == 3.0
    assert module.axis_length == 0.1


def test_articulation_link_colorer_module_defaults() -> None:
    module = ArticulationLinkColorerModule({})
    assert "palm" in module.colorer.color_map
    assert "finger_0" in module.colorer.color_map


def test_goal_visualization_is_regular_module() -> None:
    module = GoalVisualizationModule({})
    assert hasattr(module, "step")


def test_play_mode_sets_capped_step() -> None:
    env_config = {"create_rigid_vsim_envs": {"num_envs": 4096}}
    args = argparse.Namespace(
        num_envs=None,
        device=None,
        seed=None,
        mode="play",
        headless="False",
    )
    out = apply_env_overrides(env_config, args)
    assert out["create_rigid_vsim_envs"]["rendering"] is True
    assert out["create_rigid_vsim_envs"]["with_window"] is True
    assert out["render"]["capped_step"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_train_two_epochs(task_root: Path, tmp_path: Path) -> None:
    env_path = task_root / "hand_cube" / "env.yaml"
    with open(env_path, "r") as f:
        env_cfg = yaml.safe_load(f)
    vsim_path = env_cfg["create_robot"]["vsim_path"]
    if not os.path.isfile(vsim_path):
        pytest.skip(f"VSIM not found: {vsim_path}")

    runner = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "optimal_morphology_rl"
        / "runners"
        / "modular_runner.py"
    )
    cmd = [
        sys.executable,
        str(runner),
        "hand_cube",
        "--mode",
        "train",
        "--num-envs",
        "64",
        "--horizon-length",
        "32",
        "--max-epochs",
        "2",
        "--headless",
        "True",
        "--seed",
        "42",
    ]
    result = subprocess.run(
        cmd,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"Modular train run failed:\nSTDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

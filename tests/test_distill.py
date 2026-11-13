"""Unit tests for the teacher / student (distill) setup: the teacher observation group, the teacher run lookup of the
runner, the run config copy and the DistillAgent helpers (the training loop itself runs in the sim)."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from optimal_morphology_rl.algos.distill_agent import DistillAgent
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.observations.observation_manager_module import ObservationManagerModule
from optimal_morphology_rl.runners.modular_runner import RunConfigObserver, add_teacher, is_student_run, plain_config, task_configs_raw
from optimal_morphology_rl.utils.config import overlay_config

TAU_F_MAX = torch.tensor([[0.01, 0.02], [0.03, 0.04]])


def _env():
    container = ModuleContainer()
    container.device, container.total_num_envs, container.curriculum_factor = torch.device("cpu"), 2, 1.0
    container.spool = SimpleNamespace(model_tendon_indices=[0, 1], dof_indices=torch.tensor([3, 4]), pully_radius=torch.tensor([0.005, 0.005]))
    container.robot = SimpleNamespace(get_joint_pos_buf=torch.zeros(2, 5), get_joint_vel_buf=torch.zeros(2, 5))
    container.spool.slider_velocity = lambda container: torch.zeros(2, 2)
    m5 = SimpleNamespace(last_signals={"tau_f_max": TAU_F_MAX})
    env = SimpleNamespace(device=torch.device("cpu"), total_num_envs=2,
                          module_manager=SimpleNamespace(container=container, get=lambda name: m5))
    container.env = env
    return env, container


def _manager(config):
    env, container = _env()
    manager = ObservationManagerModule(config)
    manager.finalize(container)
    manager.post_finalize(container)
    env.obs_buf = container.obs_buf
    container.reset_buf = torch.ones(2, dtype=torch.bool)
    manager.reset(container)
    manager.step(container)
    return env, container, manager


def test_teacher_group_is_built_next_to_the_policy_observation() -> None:
    teacher = {"num_hist": 3, "modules": {"spool_state": {}, "spool_torques": {}}, "privileged_modules": {"spool_torques": {}}}
    env, container, _ = _manager({"num_hist": 2, "modules": {"spool_state": {}}, "teacher": teacher})
    assert env.observation_space.shape == (2 * 4,)  # spool_state: angle + velocity per spool
    assert env.teacher_observation_space.shape == (3 * 6,)  # + tau_f_max per spool, in every history frame
    assert container.obs_buf.shape == (2, 8) and container.teacher_obs_buf.shape == (2, 18)
    # The newest frame (first in the history) holds tau_f_max after the spool state; its privileged group is not built.
    assert torch.equal(container.teacher_obs_buf[:, 4:6], TAU_F_MAX)
    assert container.get("priv_obs_buf") is None


def test_teacher_group_has_its_own_noise() -> None:
    """The student's observation noise (here white angle noise) does not reach the clean teacher group."""
    student = {"spool_state": {"noise": {"angle_std": 0.5}}}
    env, container, _ = _manager({"modules": student, "teacher": {"modules": {"spool_state": {}}}})
    assert container.obs_buf[:, :2].abs().sum() > 0.0  # noisy angles (the true ones are 0)
    assert torch.all(container.teacher_obs_buf == 0.0)


def test_no_teacher_group_without_a_teacher() -> None:
    env, container, manager = _manager({"modules": {"spool_state": {}}})
    assert manager.teacher is None and container.get("teacher_obs_buf") is None
    assert not hasattr(env, "teacher_observation_space")


def _teacher_run(tmp_path: Path, frame_skip=4, with_checkpoint=True) -> Path:
    run = tmp_path / "2026_11_09_00_00_00"
    (run / "config").mkdir(parents=True)
    (run / "nn").mkdir()
    env = {"create_rigid_vsim_envs": {"timestep": 0.005, "frame_skip": frame_skip},
           "observation": {"num_hist": 2, "modules": {"spool_state": {}, "spool_torques": {}},
                           "privileged_modules": {"spool_torques": {}}},
           "reward": {"force_tracking_reward": {"normalize": 0.3}}}
    ppo = {"params": {"algo": {"name": "a2c_continuous_pink"}, "model": {"name": "continuous_a2c_logstd"},
                      "network": {"name": "actor_critic"}, "config": {"normalize_input": True}}}
    (run / "config" / "env.yaml").write_text(yaml.safe_dump(env))
    (run / "config" / "ppo.yaml").write_text(yaml.safe_dump(ppo))
    if with_checkpoint:
        (run / "nn" / f"{run.name}.pth").write_bytes(b"")
    return run


def _student(run: Path, algo="distill"):
    env = {"create_rigid_vsim_envs": {"timestep": 0.005, "frame_skip": 4},
           "observation": {"num_hist": 2, "modules": {"spool_state": {"noise": {"angle_std": 0.1}}}},
           "reward": {"force_tracking_reward": {"normalize": 0.3}}}
    ppo = {"params": {"algo": {"name": algo}, "config": {"distill": {"teacher": str(run)}}}}
    return env, ppo


def test_add_teacher_takes_the_teacher_runs_observation_and_checkpoint(tmp_path: Path) -> None:
    run = _teacher_run(tmp_path)
    env, ppo = add_teacher(*_student(run))
    teacher = env["observation"]["teacher"]
    assert teacher["modules"] == {"spool_state": {}, "spool_torques": {}} and teacher["num_hist"] == 2
    assert env["observation"]["modules"] == {"spool_state": {"noise": {"angle_std": 0.1}}}  # the student's own
    distill = ppo["params"]["config"]["distill"]
    assert distill["_teacher_checkpoint"] == str(run / "nn" / f"{run.name}.pth")
    assert distill["_teacher_params"]["network"] == {"name": "actor_critic"}


def test_add_teacher_rejects_another_control_step(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frame_skip"):
        add_teacher(*_student(_teacher_run(tmp_path, frame_skip=20)))


def test_add_teacher_needs_the_distill_algo_and_the_run_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="algo"):
        add_teacher(*_student(_teacher_run(tmp_path), algo="a2c_continuous_pink"))
    with pytest.raises(FileNotFoundError, match="missing"):
        add_teacher(*_student(_teacher_run(tmp_path / "b", with_checkpoint=False)))


def test_teacher_paths_from_a_run_or_a_checkpoint() -> None:
    run = Path("runs/adapt_force/2026_11_09_13_21_20")
    assert DistillAgent.teacher_paths(run) == (run / "nn" / f"{run.name}.pth", run)
    ckpt = run / "nn" / "last_adapt_force_ep_856.pth"
    assert DistillAgent.teacher_paths(ckpt) == (ckpt, run)


def test_distill_rejects_unknown_keys() -> None:
    """The removed teacher-driving schedule (the student always drives) fails loudly instead of being ignored."""
    params = {"config": {"distill": {"teacher": "t", "_teacher_checkpoint": "t.pth", "_teacher_params": {},
                                     "teacher_action_prob": 1.0}}}
    with pytest.raises(ValueError, match="teacher_action_prob"):
        DistillAgent("run", params)


def test_run_config_observer_copies_the_config_files(tmp_path: Path) -> None:
    (tmp_path / "env_student.yml").write_text("a: 1\n")
    (tmp_path / "dagger.yml").write_text("params: {algo: {name: distill}}\n")
    observer = RunConfigObserver({"env.yaml": tmp_path / "env_student.yml", "ppo.yaml": tmp_path / "dagger.yml"})
    run = tmp_path / "runs" / "2026_11_09_00_00_01"
    algo = SimpleNamespace(experiment_dir=str(run), games_to_track=10, ppo_device="cpu", writer=None)
    observer.after_init(algo)
    assert (run / "config" / "env.yaml").read_text() == "a: 1\n"
    # A merged (dict) config is written as YAML.
    RunConfigObserver({"ppo.yaml": {"params": {"algo": {"name": "distill"}}}}).after_init(algo)
    assert yaml.safe_load((run / "config" / "ppo.yaml").read_text()) == {"params": {"algo": {"name": "distill"}}}
    (run / "nn").mkdir()
    assert is_student_run(str(run / "nn" / "x.pth"))


def test_student_run_keeps_its_overlays_and_base_files(tmp_path: Path) -> None:
    task = tmp_path / "task"
    task.mkdir()
    (task / "env.yaml").write_text("a: 1\nb: 2\n")
    (task / "env_student.yml").write_text("# student\nb: 3\n")
    (task / "ppo.yaml").write_text("params: {algo: {name: a2c_continuous}}\n")
    (task / "dagger.yml").write_text("params: {algo: {name: distill}}\n")
    files = task_configs_raw("task", tmp_path, student=True)
    run = tmp_path / "runs" / "2026_11_09_00_00_02"
    algo = SimpleNamespace(experiment_dir=str(run), games_to_track=10, ppo_device="cpu", writer=None)
    RunConfigObserver(files).after_init(algo)
    saved = run / "config"
    assert sorted(p.name for p in saved.iterdir()) == ["dagger.yml", "env.yaml", "env_base.yaml", "env_student.yml",
                                                        "ppo.yaml", "ppo_base.yaml"]
    assert yaml.safe_load((saved / "env.yaml").read_text()) == {"a": 1, "b": 3}  # merged: what the run used
    assert (saved / "env_student.yml").read_text() == "# student\nb: 3\n"  # the overlay as it was
    assert (saved / "env_base.yaml").read_text() == "a: 1\nb: 2\n"
    assert is_student_run(str(run / "nn" / "x.pth"))
    # A PPO run keeps env.yaml / ppo.yaml only.
    assert set(task_configs_raw("task", tmp_path)) == {"env.yaml", "ppo.yaml"}


def test_plain_config_keeps_only_yaml_types() -> None:
    config = {"path": Path("/a/b"), "pair": (1, 2.0), "array": torch.tensor([0.5, 1.0]), 3: None, "nested": [{"x": True}]}
    plain = plain_config(config)
    assert plain == {"path": "/a/b", "pair": [1, 2.0], "array": [0.5, 1.0], "3": None, "nested": [{"x": True}]}
    assert yaml.safe_load(yaml.safe_dump(plain)) == plain


def test_is_student_run_false_for_ppo_runs_and_runs_without_config(tmp_path: Path) -> None:
    run = _teacher_run(tmp_path)
    assert not is_student_run(str(run / "nn" / f"{run.name}.pth"))
    assert not is_student_run(str(tmp_path / "other" / "nn" / "x.pth"))
    assert not is_student_run(None)


def test_overlay_config_merges_replaces_and_deletes() -> None:
    base = {"params": {"algo": {"name": "ppo"}, "config": {"lr": 3e-4, "levels": [0.5, 1.0], "critic": {"units": [64]},
                                                           "horizon": "${eval:'32'}"}}}
    over = {"params": {"algo": {"name": "distill"}, "config": {"levels": [1.0], "critic": None, "distill": {"teacher": "t"}}}}
    merged = overlay_config(base, over)
    assert merged == {"params": {"algo": {"name": "distill"}, "config": {"lr": 3e-4, "levels": [1.0], "horizon": "${eval:'32'}",
                                                                         "distill": {"teacher": "t"}}}}
    assert base["params"]["config"]["critic"] == {"units": [64]}  # inputs unchanged


def test_adapt_force_student_examples_load() -> None:
    from optimal_morphology_rl.runners.modular_runner import DEFAULT_TASK_ROOT, load_task_configs

    _, env, ppo = load_task_configs("adapt_force", DEFAULT_TASK_ROOT, context={"mode": "train"}, student=True)
    _, _, teacher_ppo = load_task_configs("adapt_force", DEFAULT_TASK_ROOT, context={"mode": "train"})
    _, teacher_env, _ = load_task_configs("adapt_force", DEFAULT_TASK_ROOT, context={"mode": "train"})
    # Everything env_student.yml doesn't name comes from env.yaml.
    assert {k: v for k, v in env.items() if k != "observation"} == {k: v for k, v in teacher_env.items() if k != "observation"}
    assert env["observation"]["num_hist"] == teacher_env["observation"]["num_hist"]
    student, teacher = ppo["params"], teacher_ppo["params"]
    assert student["algo"]["name"] == "distill" and student["config"]["distill"]["teacher"]
    assert "central_value_config" not in student["config"] and student["config"]["curriculum"]["levels"] == [1.0]
    # Everything dagger.yml doesn't name comes from ppo.yaml (resolved: e.g. gamma from the control step).
    assert student["network"] == teacher["network"]
    assert student["config"]["gamma"] == teacher["config"]["gamma"]
    assert student["config"]["horizon_length"] == teacher["config"]["horizon_length"]
    with pytest.raises(FileNotFoundError, match="Teacher run"):  # the example's teacher is a placeholder
        add_teacher(env, ppo)

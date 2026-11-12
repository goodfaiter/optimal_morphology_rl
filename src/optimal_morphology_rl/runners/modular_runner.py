import os

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import argparse
import copy
import shutil
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
import yaml
from rl_games.algos_torch import players
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import VlearnAlgoObserver
from rl_games.common.ivecenv import IVecEnv
from rl_games.torch_runner import Runner
from vlearn.spaces import Box, Discrete
from vlearn.torch_utils.wrappers import NewToOldAPICompatilibity

import optimal_morphology_rl.networks.transformer_network  # noqa: F401  # registers the transformer builder with rl_games
from optimal_morphology_rl.algos.colored_noise_agent import ColoredNoiseA2CAgent
from optimal_morphology_rl.algos.distill_agent import DistillAgent
from optimal_morphology_rl.envs.modular_environment import ModularEnvironment
from optimal_morphology_rl.utils.config import load_raw_yaml, load_yaml_with_context, resolve_config

# Built-in task folders (the --task-root default).
DEFAULT_TASK_ROOT = Path(__file__).resolve().parent.parent / "envs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a modular hand-object task from per-task env/ppo configs.")
    parser.add_argument("task", help="Task name, e.g. hand_cube")
    parser.add_argument(
        "--task-root",
        type=Path,
        default=DEFAULT_TASK_ROOT,
        help="Directory containing task folders",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "play", "distill", "step", "interactive"],
        default="step",
        help="Execution mode (distill: train the student of env.yaml + env_student.yml / ppo.yaml + dagger.yml from its teacher run)",
    )
    parser.add_argument("--num-envs", type=int, help="Number of parallel environments")
    parser.add_argument("--device", type=str, help="CUDA device passed to the env, e.g. cuda:0")
    parser.add_argument(
        "--headless",
        choices=["True", "False"],
        default=None,
        help="Run without rendering",
    )
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--max-epochs", type=int, help="Maximum training epochs")
    parser.add_argument("--horizon-length", type=int, help="Number of steps per rollout")
    parser.add_argument("--learning-rate", type=float, help="PPO learning rate")
    parser.add_argument("--kl-threshold", type=float, help="PPO KL threshold")
    parser.add_argument("--experiment-name", type=str, help="Name of the experiment directory")
    parser.add_argument("--cp", type=str, help="Checkpoint path (play: the policy; train: start from it; distill: the teacher, "
                        "overriding dagger.yml's distill.teacher)")
    parser.add_argument("--vsim-path", type=str, help="Override robot VSIM file path")
    parser.add_argument(
        "--deterministic",
        choices=["True", "False"],
        default=None,
        help="Deterministic actions during play",
    )
    parser.add_argument("--games-num", type=int, help="Number of games in play mode")
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of steps for step mode",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-name",
        default=None,
        help="Weights & Biases run name; turns on --wandb (default name: the --task-root experiment folder name)",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Weights & Biases project (default: optimal_morphology_rl); turns on --wandb",
    )
    return parser.parse_args()


def str_to_bool(value: str) -> bool:
    return value.lower() in ("true", "1", "yes", "on")


def task_config_paths(task: str, task_root: Path, student: bool = False) -> tuple[Path, Path | None, Path, Path | None]:
    """The task's config files: (env.yaml, env overlay, ppo.yaml, ppo overlay). The overlays (only what differs, merged
    over the base file, see overlay_config) are the student's env_student.yml and dagger.yml, or none."""
    task_dir = task_root / task
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Task folder not found: {task_dir}")
    env_path, ppo_path = task_dir / "env.yaml", task_dir / "ppo.yaml"
    env_overlay = task_dir / "env_student.yml" if student else None
    ppo_overlay = task_dir / "dagger.yml" if student else None
    for path, what in ((env_path, "env config"), (env_overlay, "student env config"), (ppo_path, "ppo config"),
                       (ppo_overlay, "dagger config")):
        if path is not None and not path.exists():
            raise FileNotFoundError(f"Missing {what}: {path}")
    return env_path, env_overlay, ppo_path, ppo_overlay


def task_configs_raw(task: str, task_root: Path, student: bool = False) -> dict[str, Path | dict[str, Any]]:
    """The run's configs as <run>/config/ keeps them: env.yaml / ppo.yaml (files), the student's merged with their
    overlays (unresolved dicts)."""
    env_path, env_overlay, ppo_path, ppo_overlay = task_config_paths(task, task_root, student)
    return {"env.yaml": load_raw_yaml(env_path, env_overlay) if env_overlay else env_path,
            "ppo.yaml": load_raw_yaml(ppo_path, ppo_overlay) if ppo_overlay else ppo_path}


def load_task_configs(
    task: str,
    task_root: Path,
    context: dict[str, Any] | None = None,
    student: bool = False,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    env_path, env_overlay, ppo_path, ppo_overlay = task_config_paths(task, task_root, student)
    env_config = resolve_config(load_raw_yaml(env_path, env_overlay), context=context)
    return env_path, env_config, load_ppo_config(ppo_path, env_config, context, ppo_overlay)


def load_ppo_config(
    ppo_path: Path, env_config: dict[str, Any], context: dict[str, Any] | None = None, overlay_path: Path | None = None
) -> dict[str, Any]:
    """The ppo config (with ``overlay_path`` merged in before resolving), with the env's num_envs, num_hist and
    control_dt in its context."""
    sim = env_config.get("create_rigid_vsim_envs", {})
    num_envs = sim.get("num_envs")
    num_hist = env_config.get("observation", {}).get("num_hist", 1)
    # The control step (s), so ppo.yaml can give time constants in seconds, e.g. a discount horizon:
    # gamma: ${eval:'1 - control_dt / 1.0'} (1 s at any frame_skip).
    control_dt = float(sim.get("timestep", 1.0 / 60.0)) * int(sim.get("frame_skip", 1))
    ppo_context = {**(context or {}), "num_envs": num_envs, "num_hist": num_hist, "control_dt": control_dt}
    return resolve_config(load_raw_yaml(ppo_path, overlay_path), context=ppo_context)


def add_teacher(env_config: dict[str, Any], ppo_config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """For a student (``--mode distill``): the teacher run of ``params.config.distill.teacher`` into both configs.

    The env gets the teacher's observation block (``observation.teacher``) from the run's saved config/env.yaml, as
    the teacher was trained; the agent its checkpoint (``distill._teacher_checkpoint``) and its ppo params
    (``distill._teacher_params``). Raises if the control step differs from the teacher's (its history would mean
    other times) and prints the other env sections that differ from the teacher's.
    """
    env_config, ppo_config = copy.deepcopy(env_config), copy.deepcopy(ppo_config)
    params = ppo_config.get("params", {})
    if params.get("algo", {}).get("name") != "distill":
        raise ValueError("--mode distill: dagger.yml needs 'params: {algo: {name: distill}}'.")
    distill = params.get("config", {}).get("distill") or {}
    if not distill.get("teacher"):
        raise ValueError("--mode distill: dagger.yml needs params.config.distill.teacher (the teacher run).")
    checkpoint, run = DistillAgent.teacher_paths(distill["teacher"])
    config_dir = run / "config"
    for path in (checkpoint, config_dir / "env.yaml", config_dir / "ppo.yaml"):
        if not path.exists():
            raise FileNotFoundError(f"Teacher run: missing {path}")
    teacher_env = load_yaml_with_context(config_dir / "env.yaml", context={"mode": "train"})
    teacher_ppo = load_ppo_config(config_dir / "ppo.yaml", teacher_env, context={"mode": "train"})

    sim, teacher_sim = env_config.get("create_rigid_vsim_envs", {}), teacher_env.get("create_rigid_vsim_envs", {})
    for key in ("timestep", "frame_skip"):
        if sim.get(key) != teacher_sim.get(key):
            raise ValueError(f"create_rigid_vsim_envs.{key}: student {sim.get(key)}, teacher {teacher_sim.get(key)} "
                             f"({config_dir / 'env.yaml'}); the teacher's history needs its control step.")
    differ = sorted(k for k in set(env_config) | set(teacher_env) if k != "observation" and env_config.get(k) != teacher_env.get(k))
    if differ:
        print(f"Student env differs from the teacher's ({config_dir / 'env.yaml'}) in: {', '.join(differ)}")

    env_config.setdefault("observation", {})["teacher"] = copy.deepcopy(teacher_env["observation"])
    distill["_teacher_checkpoint"] = str(checkpoint)
    distill["_teacher_params"] = teacher_ppo["params"]
    params["config"]["distill"] = distill
    return env_config, ppo_config


def is_student_run(checkpoint: str | None) -> bool:
    """Whether ``checkpoint`` belongs to a distill run (its saved config/ppo.yaml has algo distill)."""
    if not checkpoint:
        return False
    path = Path(checkpoint)
    run = path.parent.parent if path.parent.name == "nn" else path.parent
    ppo = run / "config" / "ppo.yaml"
    if not ppo.exists():
        return False
    algo = ((yaml.safe_load(ppo.read_text()) or {}).get("params") or {}).get("algo") or {}
    return algo.get("name") == "distill"


def apply_env_overrides(env_config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    env_config = copy.deepcopy(env_config)
    sim = env_config.setdefault("create_rigid_vsim_envs", {})

    if args.num_envs is not None:
        sim["num_envs"] = args.num_envs
    if args.device is not None:
        sim["device"] = args.device
    if args.seed is not None:
        sim["seed"] = args.seed

    headless = args.headless
    if headless is None:
        headless = "False" if args.mode in ("play", "interactive") else "True"

    # Single source of truth: --headless drives all rendering settings.
    rendering = not str_to_bool(headless)
    sim["rendering"] = rendering
    sim["with_window"] = rendering
    sim["raise_exception"] = rendering

    render_cfg = env_config.setdefault("render", {})
    render_cfg["capped_step"] = rendering
    render_cfg["paused"] = False

    if args.vsim_path is not None:
        env_config.setdefault("create_robot", {})["vsim_path"] = args.vsim_path

    return env_config


def apply_ppo_overrides(ppo_config: dict[str, Any], args: argparse.Namespace, task: str) -> dict[str, Any]:
    ppo_config = copy.deepcopy(ppo_config)
    params = ppo_config.setdefault("params", {})
    cfg = params.setdefault("config", {})

    if args.seed is not None:
        params["seed"] = args.seed
    if args.num_envs is not None:
        cfg["num_actors"] = args.num_envs
    if args.max_epochs is not None:
        cfg["max_epochs"] = args.max_epochs
    if args.horizon_length is not None:
        cfg["horizon_length"] = args.horizon_length
    if args.learning_rate is not None:
        cfg["learning_rate"] = args.learning_rate
    if args.kl_threshold is not None:
        cfg["kl_threshold"] = args.kl_threshold
    if args.experiment_name is not None:
        cfg["full_experiment_name"] = args.experiment_name

    cfg.setdefault("player", {})["use_vecenv"] = True

    if args.mode == "play":
        # Default to 1 actor in play mode (the render menu's Reset then restarts its episode timer
        # from 0; with several envs a full reset staggers the timers); user override wins.
        if args.num_envs is None:
            cfg["num_actors"] = 1
        if args.games_num is not None:
            cfg["player"]["games_num"] = args.games_num
        if args.deterministic is not None:
            cfg["player"]["deterministic"] = str_to_bool(args.deterministic)

    return ppo_config


def sync_num_envs(env_config: dict[str, Any], ppo_config: dict[str, Any]) -> dict[str, Any]:
    env_config = copy.deepcopy(env_config)
    num_actors = ppo_config.get("params", {}).get("config", {}).get("num_actors")
    if num_actors is not None:
        env_config.setdefault("create_rigid_vsim_envs", {})["num_envs"] = num_actors
    return env_config


def sync_seed(env_config: dict[str, Any], ppo_config: dict[str, Any]) -> dict[str, Any]:
    """Make ``params.seed`` of the ppo config the one seed of the run.

    rl_games seeds torch / numpy / random with it when it loads the config; copying it to
    ``create_rigid_vsim_envs.seed`` re-seeds them identically when the env is built and seeds the
    vlearn simulator, so every randomization module draws the same values for the same seed.
    """
    env_config = copy.deepcopy(env_config)
    seed = ppo_config.get("params", {}).get("seed")
    if seed is not None:
        sim = env_config.setdefault("create_rigid_vsim_envs", {})
        if sim.get("seed") not in (None, seed):
            print(f"Overriding create_rigid_vsim_envs seed {sim['seed']} with the ppo params seed {seed}.")
        sim["seed"] = int(seed)
    return env_config


def sync_curriculum_start(env_config: dict[str, Any], ppo_config: dict[str, Any]) -> dict[str, Any]:
    """Hand the curriculum's first level to the env (``_initial_curriculum_factor``), so its first episodes use it
    rather than the modules' default 1.0 (rl_games sets the factor only from its first epoch on)."""
    env_config = copy.deepcopy(env_config)
    levels = (ppo_config.get("params", {}).get("config", {}).get("curriculum") or {}).get("levels")
    if levels:
        env_config["_initial_curriculum_factor"] = float(levels[0])
    return env_config


def sync_games_to_track(env_config: dict[str, Any], ppo_config: dict[str, Any]) -> dict[str, Any]:
    """Hand the agent's ``games_to_track`` to the termination manager (``termination._games_to_track``), so the
    termination end rates (drop_rate, away_rate) average over as many ended episodes as the mean reward."""
    env_config = copy.deepcopy(env_config)
    games = ppo_config.get("params", {}).get("config", {}).get("games_to_track")
    if games is not None and "termination" in env_config:
        env_config["termination"]["_games_to_track"] = int(games)
    return env_config


def adjust_minibatch_size(config_dict: dict, num_envs: int, horizon_len: int) -> None:
    mb_size = config_dict["minibatch_size"]
    batch_size = horizon_len * num_envs
    num_batches = (batch_size + mb_size - 1) // mb_size
    if num_batches > 1:
        mb_size = batch_size // num_batches
    else:
        mb_size = batch_size

    if (batch_size % mb_size) != 0:
        raise ValueError(
            f"Batch size ({batch_size}) is not divisible by minibatch size ({mb_size}). "
            f"Batch size = horizon_length ({horizon_len}) x num_envs ({num_envs})."
        )
    config_dict["minibatch_size"] = mb_size


def convert_space(space):
    if isinstance(space, Box):
        return gym.spaces.Box(low=space.low, high=space.high, shape=space.shape)
    if isinstance(space, Discrete):
        return gym.spaces.Discrete(n=space.n)
    raise TypeError(f"Unsupported space type: {type(space)}")


class VlearnEnv(IVecEnv):
    def __init__(self, config_dict, config_name, num_actors, **kwargs):
        self.envs = config_dict[config_name]["env_creator"](num_actors, **kwargs)
        self.num_actors = num_actors

    def step(self, actions):
        return self.envs.step(actions)

    def reset(self):
        return self.envs.reset()

    def set_train_info(self, env_frames, *args, **kwargs):
        """rl_games hands the agent each epoch: pass its curriculum factor on to the modules when it changes."""
        factor = float(getattr(args[0], "curriculum_factor", 1.0)) if args else 1.0
        if factor != getattr(self, "_curriculum_factor", None):
            self._curriculum_factor = factor
            self.envs.env.module_manager.set_curriculum_factor(factor)

    def get_env_info(self):
        env_info = {
            "observation_space": convert_space(self.envs.observation_space),
            "action_space": convert_space(self.envs.action_space),
        }
        if hasattr(self.envs, "state_space"):
            env_info["state_space"] = convert_space(self.envs.state_space)
        # Read by rl_games when it saves the deployment .pt (params.config.export keys take precedence).
        env_info["export"] = export_info(self.envs.env)
        return env_info


def make_env_creator(env_config: dict[str, Any], mode: str):
    def create_envs(num_envs: int, **kwargs):
        assert torch.cuda.is_available(), "CUDA required"

        config = copy.deepcopy(env_config)
        sim = config.setdefault("create_rigid_vsim_envs", {})
        sim["num_envs"] = num_envs

        for key in ("device", "rendering", "seed"):
            if key in kwargs:
                sim[key] = kwargs.pop(key)
        if kwargs:
            print(f"Ignoring extra env creator kwargs: {sorted(kwargs.keys())}")

        env = ModularEnvironment(config)
        if mode == "play" and hasattr(env, "inference_mode_post_init_callback"):
            env.inference_mode_post_init_callback()
        return NewToOldAPICompatilibity(env)

    return create_envs


def export_info(env: ModularEnvironment) -> dict[str, Any]:
    """Deployment parameters of a built env for rl_games' policy export.

    Control frequency, observation history layout and the per-action command bounds that
    ``process_variable_actions`` maps the [-1, 1] policy actions to.
    """
    manager = env.module_manager
    container = manager.container
    info: dict[str, Any] = {"frequency": 1.0 / (container.timestep * container.frame_skip)}

    if "observation" in manager:
        obs_module = manager.get("observation")
        info["num_hist"] = obs_module.num_hist
        info["hist_stride"] = obs_module.hist_stride
        info["observation_modules"] = list(obs_module.config.get("modules", []))

    spool = getattr(container, "spool", None)
    if spool is not None:
        # How a held action is applied between policy steps ('create_spool' hold), for the robot's controller.
        info["hold"] = spool.hold

    if "apply_m5_forces" in manager and manager.get("apply_m5_forces").friction_model_name is not None:
        # The learned friction envelope the robot's controller must run for tau_f_max.
        info["friction_model"] = manager.get("apply_m5_forces").friction_model_name

    if "process_variable_actions" in manager:
        actions = manager.get("process_variable_actions")
        num_actions = env.action_space.shape[0]
        lower = torch.full((num_actions,), -1.0)
        upper = torch.full((num_actions,), 1.0)
        lower[container.root_slice] = actions.min_velocity.cpu()
        upper[container.root_slice] = actions.max_velocity.cpu()
        lower[container.active_dof_slice] = actions.min_dof_scale.cpu()
        upper[container.active_dof_slice] = actions.max_dof_scale.cpu()
        info["action_lower"] = lower.tolist()
        info["action_upper"] = upper.tolist()
    return info


def register_colored_noise_algo(runner: Runner) -> None:
    """Register the colored-noise PPO agent under the ``a2c_continuous_pink`` algo name.

    Training uses ``ColoredNoiseA2CAgent``; play mode falls back to the standard
    player so evaluation stays noise-free.
    """
    runner.algo_factory.register_builder("a2c_continuous_pink", lambda **kwargs: ColoredNoiseA2CAgent(**kwargs))
    runner.player_factory.register_builder("a2c_continuous_pink", lambda **kwargs: players.PpoPlayerContinuous(**kwargs))
    # The student of --mode distill (supervised from its teacher), played as any continuous policy.
    runner.algo_factory.register_builder("distill", lambda **kwargs: DistillAgent(**kwargs))
    runner.player_factory.register_builder("distill", lambda **kwargs: players.PpoPlayerContinuous(**kwargs))


class RunConfigObserver(VlearnAlgoObserver):
    """Also writes the run's configs into <run>/config/ (as env.yaml / ppo.yaml) once the run folder exists: a file
    is copied, a dict (an unresolved config, e.g. ppo.yaml with dagger.yml merged in) is written as YAML."""

    def __init__(self, files: dict[str, Path | dict[str, Any]] | None = None):
        super().__init__()
        self.files = files or {}

    def after_init(self, algo):
        super().after_init(algo)
        if not self.files:
            return
        config_dir = Path(algo.experiment_dir) / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        for name, source in self.files.items():
            if isinstance(source, dict):
                (config_dir / name).write_text(yaml.safe_dump(source, sort_keys=False))
            else:
                shutil.copyfile(source, config_dir / name)


def run_rl_games(
    env_config: dict[str, Any],
    ppo_config: dict[str, Any],
    mode: str,
    checkpoint: str | None,
    experiment_name: str | None = None,
    wandb: bool = False,
    wandb_name: str | None = None,
    wandb_project: str | None = None,
    config_files: dict[str, Path | dict[str, Any]] | None = None,
) -> None:
    """``mode`` train / distill (train the student) / play. ``config_files``: copied into <run>/config/ (training)."""
    cfg = ppo_config["params"]["config"]
    env_name = cfg["env_name"]
    num_envs = cfg["num_actors"]
    horizon_len = cfg["horizon_length"]

    adjust_minibatch_size(cfg, num_envs, horizon_len)
    if "central_value_config" in cfg:
        adjust_minibatch_size(cfg["central_value_config"], num_envs, horizon_len)

    env_configurations.register(
        env_name,
        {
            "vecenv_type": "VLEARN",
            "env_creator": make_env_creator(env_config, mode),
        },
    )
    vecenv.register(
        "VLEARN",
        lambda config_name, num_actors, **kwargs: VlearnEnv(env_configurations.configurations, config_name, num_actors, **kwargs),
    )

    recorder = env_config.get("camera_recorder")
    if mode == "play" and recorder and recorder.get("enabled", True) and not recorder.get("output_dir"):
        # Frames of a played checkpoint go to <run>/videos/<checkpoint>_<time>, the videos to the run folder.
        stamp = time.strftime("%Y_%m_%d_%H_%M_%S")
        if checkpoint:
            ckpt = Path(checkpoint).resolve()
            run_dir = ckpt.parent.parent if ckpt.parent.name == "nn" else ckpt.parent
            recorder["output_dir"] = str(run_dir / "videos" / f"{ckpt.stem}_{stamp}")
            # The videos themselves go to the run folder: <run>/<checkpoint>_<time>_<camera>.mp4.
            recorder.setdefault("video_dir", str(run_dir))
            recorder.setdefault("video_prefix", f"{ckpt.stem}_{stamp}_")
        else:
            recorder["output_dir"] = str(Path("runs/videos").resolve() / stamp)

    if mode in ("train", "distill"):
        run_args = {"train": True, "play": False, "profile": False}
    elif mode == "play":
        run_args = {"train": False, "play": True, "profile": False}
    else:
        raise ValueError(f"Unsupported rl_games mode: {mode}")

    if checkpoint:
        run_args["checkpoint"] = checkpoint

    # rl_games writes the end-of-training reward log into this folder.
    run_args["experiment_name"] = experiment_name or "runs"

    use_wandb = mode in ("train", "distill") and wandb
    if use_wandb:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("wandb is requested but not installed.") from exc
        wandb.init(
            project=wandb_project or "optimal_morphology_rl",
            name=wandb_name,
            sync_tensorboard=True,
            config={"env_config": env_config, "ppo_config": ppo_config, "mode": mode},
            monitor_gym=True,
            save_code=True,
        )

    runner = Runner(algo_observer=RunConfigObserver(config_files if mode in ("train", "distill") else None))
    register_colored_noise_algo(runner)
    runner.load(ppo_config)
    try:
        runner.run(run_args)
    finally:
        if use_wandb:
            import wandb

            wandb.finish()


def run_step(env_config: dict[str, Any], steps: int) -> None:
    env = ModularEnvironment(env_config)
    action_shape = env.action_space.shape

    for i in range(steps):
        actions = torch.rand((env.total_num_envs,) + action_shape, device=env.device) * 2.0 - 1.0
        obs, rew, term, trunc, info = env.step(actions)
        print(f"step {i:4d}: reward_mean={rew.mean().item():.4f} term={term.sum().item()} trunc={trunc.sum().item()}")


def run_interactive(env_config: dict[str, Any]) -> None:
    """Run an interactive GUI test using sliders from ``interactive_slider_control``."""
    env = ModularEnvironment(env_config)
    action_shape = env.action_space.shape

    print("Interactive mode started. Use the menu's Reset to reset, close the render window to exit.", flush=True)

    try:
        while True:
            actions = torch.zeros((env.total_num_envs,) + action_shape, device=env.device)
            env.step(actions)
    except RuntimeError as exc:
        if "Render window was closed" in str(exc):
            print("Render window closed, exiting interactive mode.")
        else:
            raise


def main() -> None:
    args = parse_args()

    # Play/interactive mode defaults: windowed visualization.
    if args.mode in ("play", "interactive"):
        if args.headless is None:
            args.headless = "False"
        if args.num_envs is None:
            args.num_envs = 1

    if args.mode == "interactive":
        task_dir = args.task_root / args.task
        env_path = task_dir / "test_env.yaml"
        if not env_path.exists():
            raise FileNotFoundError(f"Interactive test config not found: {env_path}")

        env_config = load_yaml_with_context(env_path, context={"mode": args.mode})
        env_config = apply_env_overrides(env_config, args)
        run_interactive(env_config)
        return

    # The student's files for --mode distill, and for playing a distill run's checkpoint.
    student = args.mode == "distill" or (args.mode == "play" and is_student_run(args.cp))
    context = {"mode": "train" if args.mode == "distill" else args.mode}
    env_path, env_config, ppo_config = load_task_configs(args.task, args.task_root, context=context, student=student)
    config_files = task_configs_raw(args.task, args.task_root, student)  # the student's: merged with their overlays
    checkpoint = args.cp
    if args.mode == "distill":
        if args.cp:  # --cp names the teacher (not a start for the student: the teacher has other inputs)
            for ppo in (ppo_config, config_files["ppo.yaml"]):
                ppo.setdefault("params", {}).setdefault("config", {}).setdefault("distill", {})["teacher"] = args.cp
            checkpoint = None
        env_config, ppo_config = add_teacher(env_config, ppo_config)

    env_config = apply_env_overrides(env_config, args)
    ppo_config = apply_ppo_overrides(ppo_config, args, args.task)
    env_config = sync_num_envs(env_config, ppo_config)
    env_config = sync_seed(env_config, ppo_config)
    env_config = sync_games_to_track(env_config, ppo_config)
    env_config = sync_curriculum_start(env_config, ppo_config)

    if args.mode == "step":
        run_step(env_config, args.steps)
    else:
        run_rl_games(
            env_config,
            ppo_config,
            args.mode,
            checkpoint,
            experiment_name=args.experiment_name,
            wandb=args.wandb or args.wandb_name is not None or args.wandb_project is not None,  # a name / project implies logging
            # The wandb run name: --wandb-name, else the experiment folder of a --task-root variant.
            wandb_name=args.wandb_name or (args.task_root.resolve().name if args.task_root.resolve() != DEFAULT_TASK_ROOT else None),
            wandb_project=args.wandb_project,
            config_files=config_files,
        )


if __name__ == "__main__":
    main()

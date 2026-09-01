import os

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import argparse
import copy
from pathlib import Path
from typing import Any

import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import VlearnAlgoObserver
from rl_games.common.ivecenv import IVecEnv
from rl_games.torch_runner import Runner
from vlearn.spaces import Box, Discrete
from vlearn.torch_utils.wrappers import NewToOldAPICompatilibity
import gymnasium as gym

from optimal_morphology_rl.envs.modular_environment import ModularEnvironment
from optimal_morphology_rl.utils.config import load_yaml_with_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a modular hand-object task from per-task env/ppo configs.")
    parser.add_argument("task", help="Task name, e.g. hand_cube")
    parser.add_argument(
        "--task-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "envs",
        help="Directory containing task folders",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "play", "step"],
        default="step",
        help="Execution mode",
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
    parser.add_argument("--cp", type=str, help="Checkpoint path for play mode")
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
    return parser.parse_args()


def str_to_bool(value: str) -> bool:
    return value.lower() in ("true", "1", "yes", "on")


def load_task_configs(
    task: str,
    task_root: Path,
    context: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    task_dir = task_root / task
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Task folder not found: {task_dir}")

    env_path = task_dir / "env.yaml"
    ppo_path = task_dir / "ppo.yaml"
    if not env_path.exists():
        raise FileNotFoundError(f"Missing env config: {env_path}")
    if not ppo_path.exists():
        raise FileNotFoundError(f"Missing ppo config: {ppo_path}")

    env_config = load_yaml_with_context(env_path, context=context)

    num_envs = env_config.get("create_rigid_vsim_envs", {}).get("num_envs")
    ppo_context = {**(context or {}), "num_envs": num_envs}
    ppo_config = load_yaml_with_context(ppo_path, context=ppo_context)

    return env_path, env_config, ppo_config


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
        headless = "False" if args.mode == "play" else "True"

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
        # Default to 4 actors in play mode for visualization; user override wins.
        if args.num_envs is None:
            cfg["num_actors"] = 4
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

    def get_env_info(self):
        env_info = {
            "observation_space": convert_space(self.envs.observation_space),
            "action_space": convert_space(self.envs.action_space),
        }
        if hasattr(self.envs, "state_space"):
            env_info["state_space"] = convert_space(self.envs.state_space)
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


def run_rl_games(
    env_config: dict[str, Any],
    ppo_config: dict[str, Any],
    mode: str,
    checkpoint: str | None,
    experiment_name: str | None = None,
    wandb: bool = False,
) -> None:
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

    if mode == "train":
        run_args = {"train": True, "play": False, "profile": False}
    elif mode == "play":
        run_args = {"train": False, "play": True, "profile": False}
    else:
        raise ValueError(f"Unsupported rl_games mode: {mode}")

    if checkpoint:
        run_args["checkpoint"] = checkpoint

    # rl_games uses this folder when freezing the traced player at the end of training.
    run_args["experiment_name"] = experiment_name or "runs"

    use_wandb = mode == "train" and wandb
    if use_wandb:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("wandb is requested but not installed.") from exc
        wandb.init(
            project="optimal_morphology_rl",
            sync_tensorboard=True,
            config={"env_config": env_config, "ppo_config": ppo_config, "mode": mode},
            monitor_gym=True,
            save_code=True,
        )

    runner = Runner(algo_observer=VlearnAlgoObserver())
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


def main() -> None:
    args = parse_args()

    # Play mode defaults: windowed, 4 parallel environments for visualization.
    if args.mode == "play":
        if args.headless is None:
            args.headless = "False"
        if args.num_envs is None:
            args.num_envs = 4

    _, env_config, ppo_config = load_task_configs(args.task, args.task_root, context={"mode": args.mode})

    env_config = apply_env_overrides(env_config, args)
    ppo_config = apply_ppo_overrides(ppo_config, args, args.task)
    env_config = sync_num_envs(env_config, ppo_config)

    if args.mode == "step":
        run_step(env_config, args.steps)
    else:
        run_rl_games(
            env_config,
            ppo_config,
            args.mode,
            args.cp,
            experiment_name=args.experiment_name,
            wandb=args.wandb,
        )


if __name__ == "__main__":
    main()

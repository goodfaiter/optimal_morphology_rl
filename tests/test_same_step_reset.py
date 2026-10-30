"""ModularEnvironment.step restarts ended episodes in the same step (gym autoreset): the step returns done with the
new episode's first observation, the terminal state's reward (with its termination penalty) and time_outs, and the
observation history takes exactly one frame per step. Runs the adapt_cube env in a subprocess (one vlearn gym per
process)."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent
ENV_YAML = REPO / "src" / "optimal_morphology_rl" / "envs" / "adapt_cube" / "env.yaml"

SCRIPT = textwrap.dedent(f'''
    from pathlib import Path
    from types import SimpleNamespace

    import torch

    from optimal_morphology_rl.envs.modular_environment import ModularEnvironment
    from optimal_morphology_rl.runners.modular_runner import apply_env_overrides
    from optimal_morphology_rl.utils.config import load_yaml_with_context

    N = 4
    cfg = load_yaml_with_context(Path("{ENV_YAML}"), context={{"mode": "play"}})
    cfg = apply_env_overrides(cfg, SimpleNamespace(num_envs=N, device=None, seed=None, vsim_path=None, mode="play", headless="True"))
    cfg["camera_recorder"]["enabled"] = False
    cfg["termination"] = {{"progress": {{"max_episode_length": 12}},
                          "drop_termination": {{"threshold": -1e6, "reward_scale": -40.0}}}}
    env = ModularEnvironment(cfg)
    observation = env.module_manager.get("observation")
    base, stride = observation._base_obs_dim, observation.hist_stride
    act = torch.full((N, 12), 0.5, device=env.device)

    def history(obs):  # (envs, history slots, base obs); slot 0 newest, one slot per `stride` steps
        return (obs["obs"] if isinstance(obs, dict) else obs).view(N, -1, base)

    env.reset()
    newest, since_reset, ends, checks = [], torch.zeros(N, dtype=torch.long), 0, 0
    for _ in range(40):
        obs, rew, term, trunc, info = env.step(act)
        h, done = history(obs), (term | trunc).cpu()
        newest.append(h[:, 0].clone())
        since_reset += 1
        since_reset[done] = 0
        for e in range(N):
            if done[e]:  # the new episode's first observation: the history holds only it
                ends += 1
                assert h[e, 1:].abs().max() == 0, e
                assert bool(info["time_outs"][e]) == bool(trunc[e]), e
            elif since_reset[e] >= stride and len(newest) > stride:  # one frame per step: slot 1 is `stride` steps old
                checks += 1
                assert torch.allclose(h[e, 1], newest[-1 - stride][e], atol=1e-6), e
    assert ends >= 8 and checks >= 40, (ends, checks)

    drop = next(m for m in env.module_manager.get("termination")._termination_modules if type(m).__name__ == "DropTermination")
    drop.config["threshold"] = 1e6  # the cube counts as dropped at any height: every episode ends in every step
    for _ in range(3):
        obs, rew, term, trunc, info = env.step(act)
        assert term.all() and history(obs)[:, 1:].abs().max() == 0
        assert (rew < -39.0).all(), rew  # the terminal state's reward, with the -40 termination penalty
    print("OK")
''')


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_same_step_reset() -> None:
    vsim = yaml.safe_load(ENV_YAML.read_text())["create_robot"]["vsim_path"]
    if not Path(vsim).is_file():
        pytest.skip(f"VSIM not found: {vsim}")
    result = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=600)
    assert result.returncode == 0 and "OK" in result.stdout, result.stdout[-2000:] + result.stderr[-4000:]


@pytest.mark.parametrize(
    "modules, error",
    [
        (["update_robot", "reward", "termination", "reset_envs", "observation"], None),
        (["update_robot", "observation", "reward", "termination"], "no 'reset_envs'"),
        (["update_robot", "reset_envs", "reward", "termination", "observation"], "after 'termination'"),
        (["update_robot", "observation", "reward", "termination", "reset_envs"], "once, after 'reset_envs'"),
        (["termination", "reset_envs", "observation", "observation"], "once, after 'reset_envs'"),
        ([{"name": "termination"}, "reset_envs", "observation"], None),
    ],
)
def test_post_physics_order_check(modules, error) -> None:
    from optimal_morphology_rl.envs.modular_environment import ModularEnvironment

    config = {"modules": {"post_physics_step_modules": modules}}
    if error is None:
        ModularEnvironment._check_post_physics_order(config)
    else:
        with pytest.raises(ValueError, match=error):
            ModularEnvironment._check_post_physics_order(config)

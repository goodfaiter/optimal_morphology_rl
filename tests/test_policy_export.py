"""Tests for the history-aware TorchScript deployment export (rl_games) configured by omrl."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from rl_games.algos_torch import model_builder
from rl_games.algos_torch import policy_export
from rl_games.algos_torch.policy_export import export_history_policy
from time_series_buffer.time_series_buffer import TimeSeriesBuffer

from optimal_morphology_rl.runners.modular_runner import export_info, load_task_configs

TASK_ROOT = Path(__file__).resolve().parent.parent / "src" / "optimal_morphology_rl" / "envs"


def _build_model(ppo_config, stacked_obs_dim: int, action_dim: int) -> torch.nn.Module:
    params = ppo_config["params"]
    model = model_builder.ModelBuilder().load(params).build(
        {
            "actions_num": action_dim,
            "input_shape": (stacked_obs_dim,),
            "num_seqs": 1,
            "value_size": 1,
            "normalize_value": True,
            "normalize_input": True,
        }
    )
    # Non-trivial normalization statistics so the test also covers norm_obs.
    model.running_mean_std.running_mean.uniform_(-1.0, 1.0)
    model.running_mean_std.running_var.uniform_(0.5, 2.0)
    return model.eval()


# adapt_finger: 200 Hz, 40 frames every 10 steps, two tendon spool targets in [0, 3.3] rad.
EXPORT = {"frequency": 200.0, "num_hist": 40, "hist_stride": 10, "action_lower": [0.0, 0.0], "action_upper": [3.3, 3.3]}


def test_export_matches_env_history(tmp_path: Path) -> None:
    _, _, ppo_config = load_task_configs("adapt_finger", TASK_ROOT, context={"mode": "play"})
    config = EXPORT
    lower, upper = torch.tensor(config["action_lower"]), torch.tensor(config["action_upper"])

    num_hist, stride, obs_dim, action_dim = config["num_hist"], config["hist_stride"], 7, 2
    model = _build_model(ppo_config, obs_dim * num_hist, action_dim)

    clip_actions = ppo_config["params"]["config"]["clip_actions"]
    path = export_history_policy(model, obs_dim * num_hist, tmp_path / "policy.pt", config, clip_actions)
    extra = {"metadata.json": ""}
    policy = torch.jit.load(str(path), map_location="cpu", _extra_files=extra)
    saved = json.loads(extra["metadata.json"])

    assert policy.frequency == 200.0
    assert saved["frequency"] == 200.0 and saved["obs_dim"] == obs_dim and saved["action_dim"] == action_dim
    assert saved["action_lower"] == [0.0, 0.0] and saved["action_upper"] == [3.3, 3.3]
    assert policy.num_hist == num_hist and policy.hist_stride == stride

    # Reference: the env's TimeSeriesBuffer stacking fed to the original model.
    buffer = TimeSeriesBuffer(num_envs=1, dim=obs_dim, max_size=1 + (num_hist - 1) * stride, stride=stride)
    torch.manual_seed(0)
    for step in range(3 * num_hist * stride):
        if step == 250:  # episode reset mid-sequence
            buffer.reset_idx(torch.tensor([True]))
            policy.reset()
        obs = torch.randn(obs_dim)
        buffer.add(obs.unsqueeze(0))
        stacked = buffer.get().view(1, -1)
        with torch.no_grad():
            expected, _, _, _ = model.a2c_network({"obs": model.norm_obs(stacked), "is_train": False, "prev_actions": None})
        actual = policy(obs)
        # The env's ProcessVariableActionsModule scaling (vlearn ``scale``).
        command = 0.5 * (expected.reshape(action_dim) + 1.0) * (upper - lower) + lower
        torch.testing.assert_close(actual, command, rtol=1e-5, atol=1e-5)
        assert actual.grad_fn is None


def test_sanity_check_rejects_mismatching_export(tmp_path: Path) -> None:
    _, _, ppo_config = load_task_configs("adapt_finger", TASK_ROOT, context={"mode": "play"})
    model = _build_model(ppo_config, 7 * 40, 2)
    path = export_history_policy(model, 7 * 40, tmp_path / "policy.pt", EXPORT, clip_actions=False)

    with torch.no_grad():
        model.a2c_network.mu.bias.add_(0.5)  # the live model drifts away from the saved file
    with pytest.raises(RuntimeError, match="does not match the live model"):
        policy_export.check_exported_policy(path, model, 7 * 40)


class _FakeManager:
    def __init__(self, modules, container):
        self._modules = modules
        self.container = container

    def __contains__(self, name):
        return name in self._modules

    def get(self, name):
        return self._modules[name]


def test_export_info_reads_built_env() -> None:
    container = SimpleNamespace(timestep=1 / 200, frame_skip=1, root_slice=slice(0, 1), active_dof_slice=slice(1, 3))
    modules = {
        "observation": SimpleNamespace(num_hist=40, hist_stride=10, config={"modules": {"spool_state": {}, "goal_finger_tip": {}}}),
        "process_variable_actions": SimpleNamespace(
            min_velocity=torch.tensor([-0.5]),
            max_velocity=torch.tensor([0.5]),
            min_dof_scale=torch.tensor([0.0, 0.0]),
            max_dof_scale=torch.tensor([3.3, 3.3]),
        ),
    }
    env = SimpleNamespace(module_manager=_FakeManager(modules, container), action_space=SimpleNamespace(shape=(3,)))

    info = export_info(env)

    assert info["frequency"] == pytest.approx(200.0)
    assert info["num_hist"] == 40 and info["hist_stride"] == 10
    assert info["observation_modules"] == ["spool_state", "goal_finger_tip"]
    assert info["action_lower"] == pytest.approx([-0.5, 0.0, 0.0])
    assert info["action_upper"] == pytest.approx([0.5, 3.3, 3.3])


def test_mlp_actor_is_exported_lean_and_frozen(tmp_path: Path) -> None:
    """A plain MLP actor is exported without the critic (single env, 1-D input) and still matches the full model."""
    _, _, ppo_config = load_task_configs("adapt_finger", TASK_ROOT, context={"mode": "play"})
    obs_dim, num_hist = 7, 3
    model = _build_model(ppo_config, obs_dim * num_hist, 2)
    config = {"num_hist": num_hist, "hist_stride": 1}
    path = export_history_policy(model, obs_dim * num_hist, tmp_path / "policy.pt", config, clip_actions=False)
    extra = {"metadata.json": ""}
    policy = torch.jit.load(str(path), map_location="cpu", _extra_files=extra)
    assert json.loads(extra["metadata.json"])["network"] == "lean_actor"
    assert not any("critic" in name for name, _ in policy.named_parameters())

    policy.reset()
    frames = [torch.randn(obs_dim) for _ in range(10)]
    for k, frame in enumerate(frames):
        exported = policy(frame)
        stacked = torch.cat([frames[k - i] if k - i >= 0 else torch.zeros(obs_dim) for i in range(num_hist)])
        with torch.no_grad():
            mu = model({"obs": stacked.unsqueeze(0), "is_train": False, "prev_actions": None})["mus"].reshape(-1)
        assert torch.allclose(exported, mu, atol=1e-5)
    policy.reset()
    assert torch.count_nonzero(policy.history) == 0

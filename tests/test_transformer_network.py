"""Unit tests for the transformer_actor_critic network builder."""

import pytest
import torch

from optimal_morphology_rl.networks.transformer_network import (
    PositionalEncoding,
    TransformerA2CBuilder,
)
from rl_games.algos_torch import model_builder
from rl_games.algos_torch.models import ModelA2CContinuousLogStd

NUM_HIST = 4
BASE_OBS = 11
ACTIONS_NUM = 2


def _params(**overrides) -> dict:
    params = {
        "name": "transformer_actor_critic",
        "separate": True,
        "space": {
            "continuous": {
                "mu_activation": "None",
                "sigma_activation": "None",
                "mu_init": {"name": "default"},
                "sigma_init": {"name": "const_initializer", "val": 0},
                "fixed_sigma": True,
            }
        },
        "transformer": {
            "history_size": NUM_HIST,
            "num_layers": 1,
            "num_heads": 4,
            "hidden_dim": 64,
            "feature_size": 16,
            "dropout": 0.0,
            "activation": "relu",
        },
    }
    for path, value in overrides.items():
        current = params
        keys = path.split("__")
        for key in keys[:-1]:
            current = current[key]
        current[keys[-1]] = value
    return params


def _build_network(params=None, input_shape=None, actions_num=ACTIONS_NUM):
    params = params if params is not None else _params()
    flat_obs = (
        params["transformer"]["history_size"] * BASE_OBS
        if input_shape is None
        else input_shape[0]
    )
    builder = model_builder.ModelBuilder().network_builder.load(params)
    assert isinstance(builder, TransformerA2CBuilder)
    return builder.build(name="a2c", actions_num=actions_num, input_shape=(flat_obs,))


def test_receives_registered_with_rl_games() -> None:
    assert "transformer_actor_critic" in model_builder.NETWORK_REGISTRY


def test_build_derives_input_size_from_obs_shape() -> None:
    net = _build_network()
    assert net.input_size == BASE_OBS
    assert net.history_size == NUM_HIST


def test_forward_shapes_with_separate_trunks() -> None:
    net = _build_network()
    obs = torch.randn(6, NUM_HIST * BASE_OBS)

    mu, sigma, value, states = net.forward({"obs": obs, "is_train": False})

    assert mu.shape == (6, ACTIONS_NUM)
    assert sigma.shape == (6, ACTIONS_NUM)
    assert value.shape == (6, 1)
    assert states is None
    # fixed_sigma const 0 -> the network returns logstd 0 everywhere.
    assert torch.allclose(sigma, torch.zeros(6, ACTIONS_NUM))
    assert net.actor_transformer is not net.critic_transformer


def test_forward_flips_newest_first_history() -> None:
    net = _build_network()
    obs = torch.randn(6, NUM_HIST * BASE_OBS)

    mu, sigma, value, _ = net.forward({"obs": obs})

    seq = obs.view(6, NUM_HIST, BASE_OBS).flip(1)
    a_out = net.actor_transformer(seq).contiguous().view(6, -1)
    c_out = net.critic_transformer(seq).contiguous().view(6, -1)
    assert torch.allclose(mu, net.mu(a_out), atol=1e-5)
    assert torch.allclose(value, net.value(c_out), atol=1e-5)


def test_model_play_and_train_paths() -> None:
    net = _build_network()
    model = ModelA2CContinuousLogStd(net)
    nn_model = model.Network(
        a2c_network=net, obs_shape=(NUM_HIST * BASE_OBS,), normalize_value=False, normalize_input=False, value_size=1
    )

    obs = torch.randn(5, NUM_HIST * BASE_OBS)
    play = nn_model.forward({"obs": obs, "is_train": False})
    assert play["actions"].shape == (5, ACTIONS_NUM)
    assert play["values"].shape == (5, 1)

    train = nn_model.forward({"obs": obs, "prev_actions": play["actions"], "is_train": True})
    assert train["mus"].shape == (5, ACTIONS_NUM)
    assert train["values"].shape == (5, 1)


def test_shared_trunk_when_separate_false() -> None:
    net = _build_network(params=_params(**{"separate": False}))
    obs = torch.randn(3, NUM_HIST * BASE_OBS)

    mu, sigma, value, _ = net.forward({"obs": obs})

    assert mu.shape == (3, ACTIONS_NUM)
    assert net.transformer is not None
    assert torch.allclose(sigma, torch.zeros(3, ACTIONS_NUM))


def test_obs_dim_not_divisible_by_history_size_raises() -> None:
    with pytest.raises(ValueError, match="divisible"):
        _build_network(params=_params(), input_shape=(NUM_HIST * BASE_OBS + 1,))


def test_heads_not_dividing_hidden_dim_raises() -> None:
    with pytest.raises(ValueError, match="num_heads"):
        _build_network(params=_params(**{"transformer__num_heads": 3}))


def test_hidden_dim_not_divisible_by_4_raises() -> None:
    with pytest.raises(ValueError, match="divisible by 4"):
        _build_network(params=_params(**{"transformer__hidden_dim": 66, "transformer__num_heads": 2}))


def test_missing_space_block_raises() -> None:
    params = _params()
    params.pop("space")
    with pytest.raises(ValueError, match="continuous action space"):
        _build_network(params=params)


def test_positional_encoding_shape_and_even_dim_guard() -> None:
    pe = PositionalEncoding(max_len=NUM_HIST, hidden_dim=64)
    x = torch.randn(2, NUM_HIST, 64)
    assert pe(x).shape == x.shape
    with pytest.raises(ValueError, match="even hidden_dim"):
        PositionalEncoding(max_len=NUM_HIST, hidden_dim=65)

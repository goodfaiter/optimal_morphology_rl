"""Transformer-based actor-critic network builder for rl_games.

Registered as ``transformer_actor_critic`` via ``model_builder.register_network``
at import time, so the ``network: name: transformer_actor_critic`` block of a
rl_games config resolves to :class:`TransformerA2CBuilder`. The config's
``transformer`` block shapes the trunk; ``history_size`` is expected to be
interpolated from the env config (``history_size: ${num_hist}`` while
``modular_runner.load_task_configs`` injects ``num_hist`` into the ppo config
context). ``input_size`` is derived at build time from the actual flat
observation shape divided by ``history_size``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from rl_games.algos_torch import model_builder
from rl_games.algos_torch import network_builder


def _get_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "elu":
        return nn.ELU()
    if name == "selu":
        return nn.SELU()
    if name == "swish":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "softplus":
        return nn.Softplus()
    if name == "None":
        return nn.Identity()
    raise ValueError(f"Unsupported activation '{name}'.")


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding over the history sequence."""

    def __init__(self, max_len: int, hidden_dim: int, device=None, dropout: float = 0.0):
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError("PositionalEncoding requires an even hidden_dim.")
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / hidden_dim)
        )
        pe = torch.zeros((max_len, hidden_dim), device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, History, Feature]
        x = x + self.pe.unsqueeze(0)
        return self.dropout(x)


class TorchTransformerModel(nn.Module):
    """Transformer trunk over the observation history.

    ``forward`` takes ``[Batch, History, Feature Dim]`` and returns
    ``[Batch, 1, output_size]``.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_layers: int,
        history_size: int,
        num_heads: int,
        hidden_dim: int,
        device=None,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()

        activation_module = _get_activation(activation)

        self.input_projection = nn.Linear(input_size, hidden_dim, device=device)
        self.positional_encoding = PositionalEncoding(
            max_len=history_size, hidden_dim=hidden_dim, device=device
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            device=device,
            dropout=dropout,
            activation=activation_module,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.output_sequence = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            _get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            _get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, output_size, device=device),
        )

        mask = nn.Transformer.generate_square_subsequent_mask(history_size).to(device)
        self.register_buffer("causal_mask", mask)

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, History, Feature Dim]
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer(x, mask=self.causal_mask, is_causal=True)
        x = x[:, -1, :]
        output = self.output_sequence(x)
        return output.unsqueeze(1)


class TransformerA2CBuilder(network_builder.NetworkBuilder):
    """rl_games network builder backed by :class:`TorchTransformerModel`.

    The observation is a flattened history (newest sample first, see
    ``TimeSeriesBuffer.get``); its time axis is flipped before the causal
    transformer so the ``x[:, -1]`` output reads the newest sample last. The
    trunk outputs a feature feeding standard mu/sigma/value heads configured
    by the ``space`` block, mirroring ``A2CBuilder.Network``.
    """

    def __init__(self, **kwargs):
        network_builder.NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return self.Network(self.params, **kwargs)

    class Network(network_builder.NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop("actions_num")
            input_shape = kwargs.pop("input_shape")
            self.value_size = kwargs.pop("value_size", 1)
            self.num_seqs = kwargs.pop("num_seqs", 1)

            network_builder.NetworkBuilder.BaseNetwork.__init__(self)

            self.separate = params.get("separate", False)
            self.value_act = self.activations_factory.create(params.get("value_activation", "None"))

            self.is_discrete = False
            self.is_multi_discrete = False
            self.is_continuous = False
            if "space" in params:
                self.is_discrete = "discrete" in params["space"]
                self.is_multi_discrete = "multi_discrete" in params["space"]
                self.is_continuous = "continuous" in params["space"]
            if not self.is_continuous:
                raise ValueError("transformer_actor_critic requires a continuous action space.")
            self.space_config = params["space"]["continuous"]
            self.fixed_sigma = self.space_config["fixed_sigma"]

            transformer_config = params.get("transformer")
            if transformer_config is None:
                raise ValueError(
                    "transformer_actor_critic requires a 'transformer' block in the rl_games network params."
                )
            self.history_size = int(transformer_config.get("history_size", 1))
            self.num_layers = int(transformer_config.get("num_layers", 2))
            self.num_heads = int(transformer_config.get("num_heads", 4))
            self.hidden_dim = int(transformer_config.get("hidden_dim", 128))
            self.feature_size = int(transformer_config.get("feature_size", self.hidden_dim // 4))
            self.dropout = float(transformer_config.get("dropout", 0.1))
            self.activation = transformer_config.get("activation", "relu")
            if self.history_size < 1:
                raise ValueError("transformer_actor_critic requires history_size >= 1.")
            if self.hidden_dim <= 0 or self.hidden_dim % self.num_heads != 0:
                raise ValueError("transformer_actor_critic requires num_heads to divide hidden_dim.")
            if self.hidden_dim % 4 != 0:
                raise ValueError("transformer_actor_critic requires hidden_dim to be divisible by 4.")
            if self.feature_size < 1:
                raise ValueError("transformer_actor_critic requires feature_size >= 1.")
            if not 0.0 <= self.dropout < 1.0:
                raise ValueError("transformer_actor_critic requires dropout in [0, 1).")

            if len(input_shape) != 1:
                raise ValueError(f"transformer_actor_critic expects a flat input_shape, got {input_shape}.")
            flat_obs_dim = int(input_shape[0])
            if flat_obs_dim % self.history_size != 0:
                raise ValueError(
                    f"Observation dim ({flat_obs_dim}) is not divisible by history_size "
                    f"({self.history_size}). history_size must match the env config's "
                    "observation.num_hist (ppo.yaml reads it via history_size: ${num_hist})."
                )
            self.input_size = flat_obs_dim // self.history_size

            trunk_args = {
                "input_size": self.input_size,
                "num_layers": self.num_layers,
                "history_size": self.history_size,
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
                "activation": self.activation,
            }
            if self.separate:
                self.actor_transformer = TorchTransformerModel(output_size=self.feature_size, **trunk_args)
                self.critic_transformer = TorchTransformerModel(output_size=self.feature_size, **trunk_args)
            else:
                self.transformer = TorchTransformerModel(output_size=self.feature_size, **trunk_args)

            self.value = nn.Linear(self.feature_size, self.value_size)
            self.mu = nn.Linear(self.feature_size, actions_num)
            self.mu_act = self.activations_factory.create(self.space_config["mu_activation"])
            self.sigma_act = self.activations_factory.create(self.space_config["sigma_activation"])

            if self.fixed_sigma:
                self.sigma = nn.Parameter(
                    torch.zeros(actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False
                )
            else:
                self.sigma = nn.Linear(self.feature_size, actions_num)

            mu_init = self.init_factory.create(**self.space_config["mu_init"])
            sigma_init = self.init_factory.create(**self.space_config["sigma_init"])
            mu_init(self.mu.weight)
            torch.nn.init.zeros_(self.mu.bias)
            if self.fixed_sigma:
                sigma_init(self.sigma)
            else:
                sigma_init(self.sigma.weight)
                torch.nn.init.zeros_(self.sigma.bias)

        def forward(self, obs_dict):
            obs = obs_dict["obs"]
            batch_size = obs.size(0)
            obs = obs.view(batch_size, self.history_size, self.input_size)
            # TimeSeriesBuffer returns the history newest-sample-first; flip so
            # the causal mask consumes the newest sample last.
            obs = obs.flip(1)

            if self.separate:
                a_out = self.actor_transformer(obs)
                c_out = self.critic_transformer(obs)
            else:
                a_out = c_out = self.transformer(obs)

            a_out = a_out.contiguous().view(batch_size, -1)
            c_out = c_out.contiguous().view(batch_size, -1)

            value = self.value_act(self.value(c_out))
            mu = self.mu_act(self.mu(a_out))
            if self.fixed_sigma:
                sigma = mu * 0.0 + self.sigma_act(self.sigma)
            else:
                sigma = self.sigma_act(self.sigma(a_out))

            return mu, sigma, value, None


model_builder.register_network("transformer_actor_critic", TransformerA2CBuilder)

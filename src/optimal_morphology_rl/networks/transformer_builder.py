import math
import torch
import torch.nn as nn

from rl_games.algos_torch.network_builder import NetworkBuilder


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, hidden_dim: int):
        super().__init__()
        position = torch.arange(seq_len).unsqueeze(1) 
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim)
        )                                             
        pe = torch.zeros(seq_len, hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, hidden_dim]
        return x + self.pe[:, : x.size(1), :]


class TransformerActorCriticBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return TransformerActorCriticBuilder.Network(self.params, **kwargs)

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop("actions_num")
            input_shape = kwargs.pop("input_shape")
            self.value_size = kwargs.pop("value_size", 1)
            self.num_seqs = kwargs.pop("num_seqs", 1)

            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)

            obs_dim = input_shape[0]
            assert obs_dim % self.seq_len == 0, (
                f"obs_dim ({obs_dim}) must be divisible by seq_len ({self.seq_len}). "
                f"Check that hist_size in the env matches seq_len in the YAML."
            )
            self.feature_dim = obs_dim // self.seq_len

            self.input_proj = nn.Linear(self.feature_dim, self.hidden_dim)
            self.pos_enc = PositionalEncoding(self.seq_len, self.hidden_dim)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_dim * 4,
                dropout=self.dropout,
                activation="relu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=self.num_layers,
                enable_nested_tensor=False,
            )

            # Causal mask
            causal_mask = nn.Transformer.generate_square_subsequent_mask(self.seq_len)
            self.register_buffer("causal_mask", causal_mask)

            self.mu_head = nn.Linear(self.hidden_dim, actions_num)
            self.value_head = nn.Linear(self.hidden_dim, self.value_size)

            if self.fixed_sigma:
                self.sigma = nn.Parameter(
                    torch.zeros(actions_num, requires_grad=True, dtype=torch.float32),
                    requires_grad=False,
                )
            else:
                self.sigma_head = nn.Linear(self.hidden_dim, actions_num)

            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
            nn.init.orthogonal_(self.value_head.weight, gain=1.0)

        def forward(self, obs_dict):
            obs = obs_dict["obs"]  # [B, seq_len * feature_dim]

            # Reshape
            B = obs.shape[0]
            x = obs.view(B, self.seq_len, self.feature_dim)  # [B, T, F]
            x = self.input_proj(x)   
            x = self.pos_enc(x)

            # Mask
            mask = self.causal_mask.to(dtype=x.dtype)
            x = self.transformer(x, mask=mask, is_causal=True)
            x = x[:, -1, :]  # [B, hidden_dim]

            mu = self.mu_head(x)
            value = self.value_head(x)

            if self.fixed_sigma:
                sigma = mu * 0.0 + self.sigma
            else:
                sigma = self.sigma_head(x)

            return mu, sigma, value, None

        def load(self, params):
            t = params["transformer"]
            self.seq_len    = t["seq_len"]
            self.hidden_dim = t["hidden_dim"]
            self.num_heads  = t["num_heads"]
            self.num_layers = t["num_layers"]
            self.dropout    = t.get("dropout", 0.0)

            space = params["space"]["continuous"]
            self.fixed_sigma = space["fixed_sigma"]

        def is_rnn(self):
            return False

        def get_default_rnn_state(self):
            return None

        def is_separate_critic(self):
            return False

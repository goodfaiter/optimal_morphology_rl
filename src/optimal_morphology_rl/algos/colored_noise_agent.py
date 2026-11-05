import numpy as np
import torch
from rl_games.algos_torch.a2c_continuous import A2CAgent

from optimal_morphology_rl.helpers.colored_noise import powerlaw_psd_gaussian


class ColoredNoiseA2CAgent(A2CAgent):
    """Continuous PPO agent with temporally correlated colored exploration noise.

    Replaces the i.i.d. sampling noise of the Gaussian policy with colored
    noise (power spectral density ~ 1/f^beta; beta=1 -> pink noise) generated
    over the time dimension of each rollout:

        action = mu + noise * sigma * scale

    Each rollout draws a new noise sequence, so by default the noise jumps between rollouts (with brown noise,
    beta = 2, a ~3 sigma kick every rollout). With ``continuous: true`` each rollout's sequence is drawn with
    ``crossfade_steps`` extra steps, and the next rollout starts exactly on that continuation and blends into its own
    new sequence over those steps (weights cos / sin, so the noise variance stays the same)::

        colored_noise:
          enabled: true
          beta: 2.0
          scale: 1.0
          continuous: true    # default false
          crossfade_steps: 64

    The noisy action is stored in the experience buffer and its log-probability
    is recomputed under the policy Gaussian, so the PPO ratio stays internally
    consistent (Eberhard et al., "Colored Noise Exploration in Deep RL", 2023).
    """

    def __init__(self, base_name, params):
        A2CAgent.__init__(self, base_name, params)
        cfg = self.config.get("colored_noise", {}) or {}
        self.colored_noise_enabled = bool(cfg.get("enabled", True))
        self.colored_noise_beta = float(cfg.get("beta", 1.0))
        self.colored_noise_scale = float(cfg.get("scale", 1.0))
        self.colored_noise_continuous = bool(cfg.get("continuous", False))
        self.colored_noise_crossfade = int(cfg.get("crossfade_steps", 64))
        if self.colored_noise_continuous and not 0 < self.colored_noise_crossfade <= self.horizon_length:
            raise ValueError(f"colored_noise crossfade_steps must be in [1, horizon_length], got {self.colored_noise_crossfade}")
        self._noise_tail = None  # the previous rollout's continuation (crossfade_steps, batch, actions)
        if self.colored_noise_enabled and self.is_rnn:
            raise NotImplementedError("Colored noise exploration is not supported for RNN models")
        self._noise_buffer = None
        self._noise_step = 0

    def _new_noise_rollout(self):
        batch = self.num_actors * self.num_agents
        if self.colored_noise_continuous:
            self._noise_buffer, self._noise_tail = self.continued_noise(
                self.colored_noise_beta, self.horizon_length, batch, self.actions_num, self.colored_noise_crossfade,
                self._noise_tail, self.ppo_device)
        else:
            shape = (self.horizon_length, batch, self.actions_num)
            self._noise_buffer = powerlaw_psd_gaussian(self.colored_noise_beta, shape, device=self.ppo_device)
        self._noise_step = 0

    @staticmethod
    def continued_noise(beta, horizon, batch, actions, crossfade, tail, device=None):
        """One rollout of colored noise that continues ``tail`` (the previous rollout's continuation, or None).

        Draws horizon + crossfade steps; the first ``crossfade`` steps blend from ``tail`` (weight cos) into the new
        sequence (weight sin), so step 0 is exactly the previous sequence's next sample. Returns (rollout noise
        (horizon, batch, actions), this sequence's continuation (crossfade, batch, actions))."""
        noise = powerlaw_psd_gaussian(beta, (horizon + crossfade, batch, actions), device=device)
        rollout, new_tail = noise[:horizon].clone(), noise[horizon:]
        if tail is not None:
            angle = torch.linspace(0.0, 0.5 * np.pi, crossfade + 1, device=noise.device)[:crossfade].view(-1, 1, 1)
            rollout[:crossfade] = torch.cos(angle) * tail + torch.sin(angle) * rollout[:crossfade]
        return rollout, new_tail

    @staticmethod
    def apply_colored_noise(res_dict, noise, scale=1.0):
        """Replace the sampled exploration noise with colored noise in-place.

        Recomputes the Gaussian neg-log-probability of the noisy action so the
        stored old logp matches the stored action.
        """
        mu = res_dict["mus"]
        sigma = res_dict["sigmas"]
        actions = mu + noise * sigma * scale
        act_dim = actions.size(-1)
        neglogp = 0.5 * (((actions - mu) / sigma) ** 2).sum(dim=-1) + 0.5 * np.log(2.0 * np.pi) * act_dim + torch.log(sigma).sum(dim=-1)
        res_dict["actions"] = actions
        res_dict["neglogpacs"] = torch.squeeze(neglogp)
        return res_dict

    def play_steps(self):
        if self.colored_noise_enabled:
            self._new_noise_rollout()
        return super().play_steps()

    def get_action_values(self, obs):
        res_dict = super().get_action_values(obs)
        if self.colored_noise_enabled and self._noise_buffer is not None:
            with torch.no_grad():
                noise = self._noise_buffer[self._noise_step]
                self.apply_colored_noise(res_dict, noise, self.colored_noise_scale)
            self._noise_step += 1
        return res_dict

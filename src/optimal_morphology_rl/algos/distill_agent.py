import time
from pathlib import Path

import torch
from rl_games.algos_torch import model_builder, torch_ext
from rl_games.algos_torch.a2c_continuous import A2CAgent


class DistillAgent(A2CAgent):
    """Student policy learned by supervised learning from a frozen teacher policy (DAgger), in the same env.

    The env builds two observations every step: the student's (``observation``, the policy input) and the teacher's
    (``observation.teacher``, as the teacher run was trained: ``container.teacher_obs_buf``; ``omrl --mode distill``
    adds it from the teacher run's saved env.yaml). The student drives the env from the start, with its mean action (no
    exploration noise: its own mistakes give the states it has to learn to recover from); each rollout step stores the
    student's observation with the teacher's mean action there as its label. The update minimizes the squared error
    between the student's and the teacher's mean actions (policy units) over the usual mini-epochs and minibatches; no
    value function, entropy or critic. The episode rewards (the best checkpoint, the curriculum's early stopping) are
    the student's own.

    Logged per epoch (distill/): the loss, the action gap (mean |student - teacher mean action| on the visited states,
    before the update), the mean reward per step and the mean |force error| (N; tasks with a force goal).

    The student is an ordinary rl_games model: checkpoints, the deployment export, play mode, the curriculum and its
    early stopping (on the episode reward) work as in PPO. ``params.config.distill`` (dagger.yml)::

        distill:
          teacher: runs/adapt_force/<teacher run>  # or a .pth of it; the run's config/ holds its env.yaml and ppo.yaml

    The runner resolves ``teacher`` to ``_teacher_checkpoint`` and ``_teacher_params`` (the teacher's ppo params:
    model, network, normalize_input / normalize_value).
    """

    def __init__(self, base_name, params):
        config = params["config"]
        if config.pop("central_value_config", None) is not None:
            print("DistillAgent: central_value_config ignored (no critic is trained)")
        distill = config.get("distill") or {}
        if "_teacher_checkpoint" not in distill or "_teacher_params" not in distill:
            raise RuntimeError("DistillAgent needs params.config.distill.teacher (resolved by 'omrl --mode distill').")
        unknown = sorted(set(distill) - {"teacher", "_teacher_checkpoint", "_teacher_params"})
        if unknown:  # e.g. the removed teacher_action_prob schedule: the student always drives
            raise ValueError(f"DistillAgent: unknown params.config.distill keys {unknown} (only 'teacher').")
        A2CAgent.__init__(self, base_name, params)
        self._distill_stats: dict[str, float] = {}

        container = self.vec_env.envs.env.module_manager.container
        if container.get("teacher_obs_buf") is None:
            raise RuntimeError("DistillAgent: the env has no teacher observation (observation.teacher).")
        self._container = container
        teacher_dim = container.teacher_obs_buf.shape[1]
        self.teacher = self._build_teacher(distill["_teacher_params"], distill["_teacher_checkpoint"], teacher_dim)
        print(f"DistillAgent: teacher {distill['_teacher_checkpoint']} ({teacher_dim} inputs), student {self.obs_shape[0]} inputs")

    def _build_teacher(self, teacher_params, checkpoint, input_dim):
        network = model_builder.ModelBuilder().load(teacher_params)
        teacher_config = teacher_params.get("config", {})
        model = network.build({
            "actions_num": self.actions_num,
            "input_shape": (input_dim,),
            "num_seqs": self.num_actors * self.num_agents,
            "value_size": self.env_info.get("value_size", 1),
            "normalize_value": teacher_config.get("normalize_value", False),
            "normalize_input": teacher_config.get("normalize_input", False),
        })
        state = torch_ext.load_checkpoint(str(checkpoint))["model"]
        try:
            model.load_state_dict(state)
        except RuntimeError as exc:
            raise RuntimeError(f"DistillAgent: the teacher checkpoint {checkpoint} does not fit its saved config with "
                               f"{input_dim} observation inputs (the teacher group of this env): {exc}") from exc
        model.to(self.ppo_device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    def teacher_mus(self):
        """The teacher's mean action on its observation of the current step."""
        obs = self._container.teacher_obs_buf
        return self.teacher({"is_train": False, "prev_actions": None, "obs": obs, "rnn_states": None})["mus"]

    def play_steps(self):
        obses = torch.zeros((self.horizon_length, self.num_actors) + self.obs_shape, device=self.ppo_device)
        labels = torch.zeros((self.horizon_length, self.num_actors, self.actions_num), device=self.ppo_device)
        step_time = 0.0
        # Sums over the rollout (on the device).
        reward_sum, error_sum, gap_sum = (torch.zeros((), device=self.ppo_device) for _ in range(3))
        has_force = self._container.get("fingertip_force") is not None and self._container.get("force_goal") is not None
        for n in range(self.horizon_length):
            student_mus = self.get_action_values(self.obs)["mus"]
            teacher_mus = self.teacher_mus()
            obses[n] = self.obs["obs"]
            labels[n] = teacher_mus
            gap_sum += (student_mus - teacher_mus).abs().mean(dim=-1).sum()

            step_start = time.perf_counter()
            self.obs, rewards, self.dones, infos = self.env_step(student_mus)
            step_time += time.perf_counter() - step_start

            reward_sum += rewards.sum()
            if has_force:
                error_sum += (self._container.fingertip_force - self._container.force_goal).abs().sum()

            self.current_rewards.add_(rewards)
            self.current_shaped_rewards.add_(self.rewards_shaper(rewards))
            self.current_lengths.add_(1)
            env_done_indices = self.dones.nonzero(as_tuple=False)[::self.num_agents]
            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_shaped_rewards.update(self.current_shaped_rewards[env_done_indices])
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)
            not_dones = (1.0 - self.dones.float()).unsqueeze(1)
            self.current_rewards.mul_(not_dones)
            self.current_shaped_rewards.mul_(not_dones)
            self.current_lengths.mul_(not_dones.squeeze(1))

        steps = self.horizon_length * self.num_actors
        stats = {"action_gap": (gap_sum / steps).item(), "step_reward": (reward_sum / steps).item()}
        if has_force:
            stats["force_error_n"] = (error_sum / steps).item()
        return {"obses": obses.reshape((-1,) + self.obs_shape), "labels": labels.reshape(-1, self.actions_num),
                "played_frames": self.batch_size, "step_time": step_time, "stats": stats}

    def train_epoch(self):
        self.vec_env.set_train_info(self.frame, self)  # A2CBase.train_epoch: the curriculum factor to the env

        self.set_eval()
        play_start = time.perf_counter()
        with torch.no_grad():
            batch = self.play_steps()
        play_end = time.perf_counter()

        self.set_train()
        self.curr_frames = batch.pop("played_frames")
        self.algo_observer.after_steps()
        obses, labels = batch["obses"], batch["labels"]
        losses = []
        for _ in range(self.mini_epochs_num):
            perm = torch.randperm(obses.shape[0], device=self.ppo_device)
            for start in range(0, obses.shape[0], self.minibatch_size):
                idx = perm[start:start + self.minibatch_size]
                result = self.model({"is_train": True, "prev_actions": labels[idx],
                                     "obs": self._preproc_obs(obses[idx]), "rnn_states": None})
                loss = ((result["mus"] - labels[idx]) ** 2).sum(dim=-1).mean()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.truncate_grads:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.optimizer.step()
                losses.append(loss.detach())
            if self.normalize_input:
                self.model.running_mean_std.eval()  # the input statistics from the first mini-epoch only, as PPO
        update_end = time.perf_counter()

        frame = self.frame // self.num_agents
        self._distill_stats = {"loss": torch_ext.mean_list(losses).item(), **batch["stats"]}
        for name, value in self._distill_stats.items():
            self.writer.add_scalar(f"distill/{name}", value, frame)
        zero = [torch.zeros((), device=self.ppo_device)]
        return (batch["step_time"], play_end - play_start, update_end - play_end, update_end - play_start,
                losses, zero, [], zero, zero, self.last_lr, 1.0)

    def status_lines(self, *args, **kwargs):
        """The usual status lines and one with the distillation's values of the last epoch."""
        lines = super().status_lines(*args, **kwargs)
        d = self._distill_stats
        if not d:
            return lines
        parts = [f"loss {d['loss']:.4f}", f"action gap {d['action_gap']:.3f}", f"reward/step {d['step_reward']:.3f}"]
        if "force_error_n" in d:
            parts.append(f"force error {d['force_error_n']:.2f} N")
        return lines + [" | ".join(parts)]

    @staticmethod
    def teacher_paths(teacher: str | Path) -> tuple[Path, Path]:
        """(checkpoint, run folder) of ``teacher``: a run folder (its best checkpoint nn/<run>.pth) or a .pth in it."""
        path = Path(teacher)
        if path.suffix == ".pth":
            run = path.parent.parent if path.parent.name == "nn" else path.parent
            return path, run
        return path / "nn" / f"{path.name}.pth", path

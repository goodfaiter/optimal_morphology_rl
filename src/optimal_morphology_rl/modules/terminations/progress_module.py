"""Termination sub-module that tracks episode progress and truncation."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.terminations.termination_base_module import (
    TerminationBaseModule,
    register_termination,
)


@register_termination("progress")
class ProgressModule(TerminationBaseModule):
    """Owns ``progress_buf`` and the time-based truncation signal.

    This module increments ``progress_buf`` every step and marks environments
    as truncated when ``progress_buf >= max_episode_length``.  It also resets
    ``progress_buf`` for environments selected by ``reset_buf``.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.max_episode_length = int(
            self.config.get("max_episode_length", 6 * 60)
        )

    def post_finalize(self, container: ModuleContainer) -> None:
        """Allocate the per-environment progress buffer."""
        if container.get("progress_buf") is not None:
            return
        if container.get("total_num_envs") is None or container.get("device") is None:
            raise RuntimeError(
                "ProgressModule requires 'total_num_envs' and 'device' in the shared container."
            )
        container.progress_buf = torch.zeros(
            container.total_num_envs, dtype=torch.long, device=container.device
        )

    def compute(self, env: Any) -> None:
        """Increment progress and apply the episode-length truncation."""
        env.progress_buf.add_(1)
        env.trunc_buf[env.progress_buf >= self.max_episode_length] = True

    def reset(self, container: ModuleContainer) -> None:
        """Reset progress for the environments selected by ``reset_buf``.

        When every environment is being reset (initial reset) we randomize
        progress to diversify episode lengths, matching the legacy reset
        behavior.  Partial resets zero progress for the affected envs.
        """
        reset_buf = container.get("reset_buf")
        progress_buf = container.get("progress_buf")
        if reset_buf is None or progress_buf is None:
            return
        if reset_buf.all() and container.get("total_num_envs", 0) > 1:
            progress_buf[:] = torch.randint(
                0,
                self.max_episode_length,
                (container.total_num_envs,),
                device=container.device,
            )
        else:
            progress_buf[reset_buf] = 0

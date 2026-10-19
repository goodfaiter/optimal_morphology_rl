"""Module that starts episodes from recorded robot joint states (a state bank loaded from an .npz)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module

ENVS_DIR = Path(__file__).resolve().parent.parent / "envs"


def resolve_states_path(path: str | Path) -> Path:
    """Absolute paths are used as is, relative ones resolve against the package's ``envs`` folder."""
    path = Path(path)
    return path if path.is_absolute() else ENVS_DIR / path


@register_module("initial_state_bank")
class InitialStateBankModule(BaseModule):
    """Randomizes the initial robot pose with recorded, physically consistent joint states.

    Randomizing joint positions directly breaks tendon-driven hands: the tendon length is set by
    the spool sliders, so a random finger pose with the sliders at their reset position leaves the
    tendons inconsistent with the spools. Instead the full joint state (finger joints and sliders)
    is recorded from a simulation run (``scripts/record_initial_states.py``) into an .npz file with
    ``joint_pos`` ``(num_states, num_joints)`` and ``joint_names``. On reset, each resetting env starts
    from a random stored state (zero velocity) with probability ``prob``, else from the zero pose.
    The tendon geometry is the same in every env (the randomizations only change masses, friction
    and servo parameters), so every stored state is valid in every env.

    List it in ``init_modules`` after 'create_robot': its reset overwrites the reset joint state
    'create_robot' just wrote and re-applies it.

    Config shape::

        initial_state_bank:
          path: adapt_finger/initial_states.npz  # relative to the package's envs folder
          prob: 0.5                              # fraction of resets that start from a stored state
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.prob = float(self.config.get("prob", 0.5))
        if not 0.0 <= self.prob <= 1.0:
            raise RuntimeError(f"InitialStateBankModule config 'prob' must be in [0, 1], got {self.prob}.")
        if "path" not in self.config:
            raise RuntimeError("InitialStateBankModule config requires 'path' (an .npz of recorded joint states).")
        self.path = resolve_states_path(self.config["path"])
        self.states: torch.Tensor | None = None

    def finalize(self, container: ModuleContainer) -> None:
        if container.get("robot") is None:
            raise RuntimeError("InitialStateBankModule requires 'create_robot' (listed before 'initial_state_bank').")

    def post_finalize(self, container: ModuleContainer) -> None:
        if not self.path.is_file():
            raise RuntimeError(f"InitialStateBankModule: '{self.path}' not found; record it with scripts/record_initial_states.py.")
        data = np.load(self.path)
        robot = container.robot
        names = list(robot.art_def.get_joint_dof_def_names())
        stored = [str(n) for n in data["joint_names"]]
        if stored != names:
            raise RuntimeError(f"InitialStateBankModule: '{self.path.name}' was recorded for joints {stored}, the robot has {names}.")
        self.states = torch.as_tensor(data["joint_pos"], dtype=torch.float32, device=container.device)
        print(f"InitialStateBankModule: {self.states.shape[0]} initial states from '{self.path.name}', prob {self.prob}")

    def reset(self, container: ModuleContainer) -> None:
        """Start a random share of the resetting envs from stored states (after 'create_robot' wrote the zero pose)."""
        if self.prob == 0.0 or self.states is None or self.states.shape[0] == 0:
            return
        reset_buf = container.reset_buf
        use_bank = reset_buf & (torch.rand(reset_buf.shape[0], device=reset_buf.device) < self.prob)
        picks = torch.randint(0, self.states.shape[0], (reset_buf.shape[0],), device=reset_buf.device)
        container.reset_joint_pos_buf[use_bank] = self.states[picks[use_bank]]
        container.reset_joint_vel_buf[use_bank] = 0.0
        # Re-apply create_robot's reset command (masked by reset_buf) with the stored rows.
        container.gym.set_articulation_kinematic_states(container.gpu_reset_kinematic_state_command_array)

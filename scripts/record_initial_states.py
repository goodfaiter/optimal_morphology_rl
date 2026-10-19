"""Record physically consistent robot joint states for 'initial_state_bank'.

Runs the task's training env (randomizations on, initial_state_bank off) with random spool targets
held for random durations, so the finger visits poses from fully open to fully curled, and every
RECORD_EVERY control steps stores the full joint state (finger joints and spool sliders) of
RECORD_COUNT random envs. Writes ``joint_pos`` (num_states, num_joints) and ``joint_names`` to OUTPUT.

Run with:  uv run python3 scripts/record_initial_states.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from optimal_morphology_rl.envs.modular_environment import ModularEnvironment
from optimal_morphology_rl.utils.config import load_yaml_with_context

TASK = "adapt_finger"
ENVS_DIR = Path(__file__).resolve().parent.parent / "src" / "optimal_morphology_rl" / "envs"
OUTPUT = ENVS_DIR / TASK / "initial_states.npz"
NUM_ENVS = 256
STEPS = 2000  # control steps (10 s at 200 Hz)
HOLD_STEPS = (20, 200)  # each env holds a random action for this many steps
WARMUP_STEPS = 100  # not recorded: lets the envs leave the reset pose
RECORD_EVERY = 10
RECORD_COUNT = 32
SEED = 0


def main() -> None:
    config = load_yaml_with_context(ENVS_DIR / TASK / "env.yaml", context={"mode": "train"})
    config["create_rigid_vsim_envs"].update(num_envs=NUM_ENVS, rendering=False, with_window=False, seed=SEED)
    for phase in ("init_modules", "post_physics_step_modules"):
        modules = config["modules"].get(phase, [])
        if "initial_state_bank" in modules:
            modules.remove("initial_state_bank")  # record from the plain reset pose
    config.get("spool_plot", {})["enabled"] = False

    env = ModularEnvironment(config)
    env.reset()
    container = env.module_manager.container
    robot, device = container.robot, container.device
    names = list(robot.art_def.get_joint_dof_def_names())

    generator = torch.Generator(device=device).manual_seed(SEED)
    action = torch.rand((NUM_ENVS,) + env.action_space.shape, device=device, generator=generator) * 2 - 1
    hold = torch.randint(*HOLD_STEPS, (NUM_ENVS,), device=device, generator=generator)
    states = []
    for step in range(STEPS):
        hold -= 1
        expired = hold <= 0
        if expired.any():
            new = torch.rand((NUM_ENVS,) + env.action_space.shape, device=device, generator=generator) * 2 - 1
            action[expired] = new[expired]
            hold[expired] = torch.randint(*HOLD_STEPS, (int(expired.sum()),), device=device, generator=generator)
        env.step(action)
        if step >= WARMUP_STEPS and step % RECORD_EVERY == 0:
            envs = torch.randint(0, NUM_ENVS, (RECORD_COUNT,), device=device, generator=generator)
            states.append(robot.get_joint_pos_buf[envs].clone())

    joint_pos = torch.cat(states).cpu().numpy()
    if not np.isfinite(joint_pos).all():
        raise RuntimeError("Recorded non-finite joint states.")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT, joint_pos=joint_pos, joint_names=np.array(names))
    print(f"Saved {joint_pos.shape[0]} states of joints {names} to {OUTPUT}")
    for i, name in enumerate(names):
        print(f"  {name:32s} [{joint_pos[:, i].min():8.4f}, {joint_pos[:, i].max():8.4f}]")


if __name__ == "__main__":
    main()

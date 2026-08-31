"""Environment implementations."""

from optimal_morphology_rl.envs.modular_environment import ModularEnvironment

# Import task packages so their reward/termination decorators register.
from optimal_morphology_rl.envs import hand_button
from optimal_morphology_rl.envs import hand_button_difficult
from optimal_morphology_rl.envs import hand_cube
from optimal_morphology_rl.envs import hand_drawer
from optimal_morphology_rl.envs import hand_tomato
from optimal_morphology_rl.envs import hand_tomato_extreme

__all__ = ["ModularEnvironment"]

"""Fingertip force task with a moving sensor: hold 1 N on a cylinder placed anywhere along the middle finger's curl path
(and moved along it during the episode); adapt_force's goal, observations and rewards (force_goal) plus the cylinder
placement (finger_path_cylinder)."""

from optimal_morphology_rl.envs.adapt_force import force_goal  # noqa: F401 (registers force_goal and its rewards)
from optimal_morphology_rl.envs.adapt_force_moving import finger_path_cylinder  # noqa: F401

"""Module that runs the simulator's forward kinematics after a physics step."""

from __future__ import annotations

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("update_kinematics")
class UpdateKinematicsModule(BaseModule):
    """Forward kinematics (``gym.compute_kinematics()``) after a physics step: kinematic sensor states and link
    transforms are only up to date after it (without it the cube sensor's linear velocity read ~0.76x the object's;
    joint and tendon states do not need it). ~0.03 ms. List it before the modules that read them: first in
    ``post_gym_step_modules`` (every physics substep) or first in ``post_physics_step_modules`` (once per control
    step, after the substeps)::

        modules:
          post_gym_step_modules: [update_kinematics, update_robot]
          post_physics_step_modules: [update_kinematic_sensor, ...]
    """

    def step(self, container: ModuleContainer) -> None:
        container.gym.compute_kinematics()

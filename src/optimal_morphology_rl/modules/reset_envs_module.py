"""Module that restarts the episodes ended in this step (gym's autoreset)."""

from __future__ import annotations

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("reset_envs")
class ResetEnvsModule(BaseModule):
    """Restart the episodes the termination modules ended in this step (``reset_buf``), as IsaacLab does.

    rl_games expects done with the new episode's first observation: its GAE does not bootstrap from the observation
    returned with done. ModularEnvironment.step returns the reward and the term / trunc flags of the state the
    physics reached (saved here as ``episode_end_term`` / ``episode_end_trunc``; the reset clears ``term_buf`` /
    ``trunc_buf``) with the observation computed after this module.

    Every module's ``reset`` handles its part for the reset envs: the ones listed first write the new state (e.g.
    create_robot, create_objects, randomize_*), the readers (update_robot, update_objects, update_kinematic_sensor)
    then read it. List it in ``post_physics_step_modules`` after ``termination`` and before ``observation``, which
    must run once per step (its history takes one frame per call)::

        post_physics_step_modules: [update_robot, ..., reward, termination, reset_envs, observation, ...]
    """

    def step(self, container: ModuleContainer) -> None:
        container.episode_end_term = container.term_buf.clone()
        container.episode_end_trunc = container.trunc_buf.clone()
        if container.reset_buf.any():
            container.get("env").module_manager.reset()

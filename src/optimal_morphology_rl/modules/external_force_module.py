"""Module that applies random external forces to the reward object."""

from __future__ import annotations

from typing import Any

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.external_force import (
    ExternalForceConfig,
    ExternalForceModule as ExternalForceHelper,
)
from optimal_morphology_rl.modules.module_manager import register_module


@register_module("external_force")
class ExternalForceModule(BaseModule):
    """Applies random external forces to the reward object.

    Only active when the reward object is a loaded rigid object that is not the
    cube (matching the legacy behavior).
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.apply_prob = float(self.config.get("apply_prob", 0.02))
        self.force_max = float(self.config.get("force_max", 2.0))
        self.torque_max = float(self.config.get("torque_max", 0.0))
        self.force_module: ExternalForceHelper | None = None

    def post_finalize(self, env: Any) -> None:
        container = env.module_manager.container
        reward_object = container.get("reward_object")
        if reward_object is None:
            return

        from optimal_morphology_rl.modules.objects import LoadedRigidObject

        reward_object_name = container.get("reward_object_name", "")
        if reward_object_name == "cube" or not isinstance(
            reward_object, LoadedRigidObject
        ):
            return

        config = ExternalForceConfig(
            apply_prob=self.apply_prob,
            force_max=self.force_max,
            torque_max=self.torque_max,
        )
        self.force_module = ExternalForceHelper(
            body_handles={reward_object_name: reward_object.handle},
            total_num_envs=env.total_num_envs,
            device=env.device,
            env_group=container.env_group,
            gym=container.gym,
            config=config,
        )
        container.external_force = self.force_module

    def pre_physics_step(self, env: Any) -> None:
        if self.force_module is not None:
            self.force_module.step(env.module_manager.container.gym)

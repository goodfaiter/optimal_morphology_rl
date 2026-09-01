"""Manager module that owns reward sub-modules and computes the reward."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import ModuleManager, register_module
from optimal_morphology_rl.modules.rewards.reward_base_module import RewardBaseModule


# Registry for reward sub-modules.
REWARD_REGISTRY: dict[str, type[RewardBaseModule]] = {}


def register_reward(name: str | None = None):
    """Decorator that registers a reward sub-module class."""

    def decorator(cls: type[RewardBaseModule]) -> type[RewardBaseModule]:
        registry_name = name if name is not None else cls.__name__
        if registry_name in REWARD_REGISTRY:
            raise ValueError(
                f"Reward module '{registry_name}' is already registered."
            )
        REWARD_REGISTRY[registry_name] = cls
        return cls

    return decorator


@register_module("reward")
class RewardManagerModule(BaseModule):
    """Computes the reward from registered reward sub-modules.

    Config shape::

        reward:
          goal_position_reward:
            scale: 1.5
          goal_orientation_reward:
            scale: 1.0

    Reward module names are discovered from the top-level keys of the
    ``reward`` block (order is preserved from the YAML).
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.sub_manager = self._build_sub_manager()
        self._reward_modules: list[RewardBaseModule] = [
            m for m in self.sub_manager if isinstance(m, RewardBaseModule)
        ]

    def _build_sub_manager(self) -> ModuleManager:
        """Create a ModuleManager for reward sub-modules from config keys."""
        sub_config: dict[str, Any] = {"modules": {"init_modules": []}}
        for key in self.config.keys():
            if key.startswith("_"):
                continue
            sub_config["modules"]["init_modules"].append(key)
            sub_config[key] = self.config[key]
        return ModuleManager.from_config(sub_config, registry=REWARD_REGISTRY)

    def post_finalize(self, container: ModuleContainer) -> None:
        """Allocate the reward buffer and run post_finalize on reward sub-modules."""
        if container.get("total_num_envs") is None or container.get("device") is None:
            raise RuntimeError(
                "RewardManagerModule requires 'total_num_envs' and 'device' "
                "in the shared container."
            )
        container.rew_buf = torch.zeros(
            container.total_num_envs, dtype=torch.float32, device=container.device
        )
        self.sub_manager.container = container
        self.sub_manager.post_finalize()

    def step(self, container: ModuleContainer) -> None:
        """Reset reward buffers and sum contributions from every reward sub-module."""
        env = container.get("env")
        if env is None:
            raise RuntimeError(
                "RewardManagerModule requires 'env' in the shared container."
            )

        env.rew_buf[:] = 0.0
        env.info["rewards"] = {}

        for module in self._reward_modules:
            contribution = module.compute(env)
            if contribution is not None:
                env.rew_buf[:] += contribution

    def reset(self, container: ModuleContainer) -> None:
        """Reset reward sub-modules."""
        self.sub_manager.container = container
        self.sub_manager.reset()

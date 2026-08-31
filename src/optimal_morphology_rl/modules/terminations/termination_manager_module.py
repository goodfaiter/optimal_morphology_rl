"""Manager module that owns termination sub-modules and computes term/trunc."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_manager import ModuleManager, register_module
from optimal_morphology_rl.modules.terminations.termination_base_module import (
    TerminationBaseModule,
)


# Registry for termination sub-modules.
TERMINATION_REGISTRY: dict[str, type[TerminationBaseModule]] = {}


def register_termination(name: str | None = None):
    """Decorator that registers a termination sub-module class."""

    def decorator(cls: type[TerminationBaseModule]) -> type[TerminationBaseModule]:
        registry_name = name if name is not None else cls.__name__
        if registry_name in TERMINATION_REGISTRY:
            raise ValueError(
                f"Termination module '{registry_name}' is already registered."
            )
        TERMINATION_REGISTRY[registry_name] = cls
        return cls

    return decorator


@register_module("termination")
class TerminationManagerModule(BaseModule):
    """Computes terminations and truncations from registered termination sub-modules.

    Config shape::

        termination:
          drop_termination:
            threshold: -0.1
          bounds_termination: {}

    Termination module names are discovered from the top-level keys of the
    ``termination`` block (order is preserved from the YAML).
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.sub_manager = self._build_sub_manager()
        self._termination_modules: list[TerminationBaseModule] = [
            m for m in self.sub_manager if isinstance(m, TerminationBaseModule)
        ]

    def _build_sub_manager(self) -> ModuleManager:
        """Create a ModuleManager for termination sub-modules from config keys."""
        sub_config: dict[str, Any] = {"modules": []}
        for key in self.config.keys():
            if key.startswith("_"):
                continue
            sub_config["modules"].append(key)
            sub_config[key] = self.config[key]
        return ModuleManager.from_config(sub_config, registry=TERMINATION_REGISTRY)

    def post_finalize(self, env: Any) -> None:
        """Run post_finalize on termination sub-modules."""
        self.sub_manager.post_finalize(env)

    def compute(self, env: Any) -> None:
        """Reset ``term_buf``, call every termination module, then compute truncation."""
        env.term_buf[:] = False

        for module in self._termination_modules:
            module.compute(env)

        env.trunc_buf[:] = env.progress_buf >= env.max_episode_length

    def reset(self, env: Any) -> None:
        """Reset termination sub-modules."""
        self.sub_manager.reset(env)

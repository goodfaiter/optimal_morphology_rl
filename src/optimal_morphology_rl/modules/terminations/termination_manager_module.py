"""Manager module that owns termination sub-modules and computes term/trunc/reset."""

from __future__ import annotations

from typing import Any

import torch

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import ModuleManager, register_module

# Import sub-modules for their @register_termination side effects.
from optimal_morphology_rl.modules.terminations import (
    bounds_termination,  # noqa: F401
    drop_termination,  # noqa: F401
    progress_module,  # noqa: F401
)
from optimal_morphology_rl.modules.terminations.termination_base_module import (
    TERMINATION_REGISTRY,
    TerminationBaseModule,
)


@register_module("termination")
class TerminationManagerModule(BaseModule):
    """Owns termination buffers and dispatches to termination sub-modules.

    Config shape::

        termination:
          progress: {}
          drop_termination:
            threshold: -0.1
          bounds_termination: {}

    Termination sub-module names are discovered from the top-level keys of the
    ``termination`` block (order is preserved from the YAML).
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.sub_manager = self._build_sub_manager()
        self._termination_modules: list[TerminationBaseModule] = [m for m in self.sub_manager if isinstance(m, TerminationBaseModule)]

    def _build_sub_manager(self) -> ModuleManager:
        """Create a ModuleManager for termination sub-modules from config keys."""
        sub_config: dict[str, Any] = {"modules": {"init_modules": []}}
        for key in self.config.keys():
            if key.startswith("_"):
                continue
            sub_config["modules"]["init_modules"].append(key)
            sub_config[key] = self.config[key]
        return ModuleManager.from_config(sub_config, registry=TERMINATION_REGISTRY)

    def finalize(self, container: ModuleContainer) -> None:
        """Allocate the termination buffers now that env size/device are known."""
        if container.get("total_num_envs") is None or container.get("device") is None:
            raise RuntimeError(
                "TerminationManagerModule requires 'total_num_envs' and 'device' "
                "in the shared container. Ensure create_rigid_vsim_envs is listed first."
            )
        total_num_envs = container.total_num_envs
        device = container.device
        container.term_buf = torch.zeros(total_num_envs, dtype=torch.bool, device=device)
        container.trunc_buf = torch.zeros(total_num_envs, dtype=torch.bool, device=device)
        container.reset_buf = torch.zeros(total_num_envs, dtype=torch.bool, device=device)
        container.inverse_reset_buf = torch.zeros(total_num_envs, dtype=torch.bool, device=device)

    def post_finalize(self, container: ModuleContainer) -> None:
        """Run post_finalize on termination sub-modules."""
        self.sub_manager.container = container
        self.sub_manager.post_finalize()

    def step(self, container: ModuleContainer) -> None:
        """Zero term/trunc, dispatch to sub-modules, then compute reset_buf."""
        env = container.get("env")
        if env is None:
            raise RuntimeError("TerminationManagerModule requires 'env' in the shared container.")

        env.term_buf[:] = False
        env.trunc_buf[:] = False

        for module in self._termination_modules:
            module.compute(env)

        container.reset_buf[:] = torch.logical_or(env.term_buf, env.trunc_buf)
        container.inverse_reset_buf[:] = ~container.reset_buf

    def reset(self, container: ModuleContainer) -> None:
        """Reset termination state for the environments selected by reset_buf."""
        reset_buf = container.get("reset_buf")
        if reset_buf is not None:
            container.term_buf[reset_buf] = False
            container.trunc_buf[reset_buf] = False
        self.sub_manager.container = container
        self.sub_manager.reset()

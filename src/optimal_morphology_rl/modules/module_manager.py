"""Module manager that loads modules from a config and dispatches lifecycle hooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, TypeVar

from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.utils.config import load_yaml_with_context


T = TypeVar("T", bound=BaseModule)

# Global registry mapping module names to module classes.
DEFAULT_REGISTRY: dict[str, type[BaseModule]] = {}


def register_module(name: str | None = None) -> Callable[[type[T]], type[T]]:
    """Decorator that registers a module class in the default registry.

    Args:
        name: Name used in YAML to refer to this module.  If ``None``, the
            class name is used.

    Returns:
        The decorated class (unchanged).
    """

    def decorator(cls: type[T]) -> type[T]:
        registry_name = name if name is not None else cls.__name__
        if registry_name in DEFAULT_REGISTRY:
            raise ValueError(f"Module '{registry_name}' is already registered ({DEFAULT_REGISTRY[registry_name].__name__}).")
        DEFAULT_REGISTRY[registry_name] = cls
        return cls

    return decorator


class ModuleManager:
    """Owns phase-grouped modules and dispatches lifecycle hooks.

    Modules are instantiated from a config dict.  The special key
    ``config["modules"]`` must contain a dict with phase keys.  Each phase
    contains an ordered list of module names.  Each name is looked up in a
    registry and receives the sub-dict at ``config[name]``.

    Example YAML::

        modules:
          init_modules:
            - create_rigid_vsim_envs
            - robot
          pre_physics_step_modules:
            - robot_control
          pre_gym_step_modules: []
          post_gym_step_modules:
            - render
          post_physics_step_modules:
            - observation
            - reward
            - termination

        robot:
          vsim_path: /path/to/hand.vsim
    """

    _PHASE_NAMES = [
        "init_modules",
        "pre_physics_step_modules",
        "pre_gym_step_modules",
        "post_gym_step_modules",
        "post_physics_step_modules",
    ]

    def __init__(
        self,
        phase_modules: dict[str, list[BaseModule]],
        name_map: dict[str, BaseModule] | None = None,
    ):
        """Create a manager from pre-built phase module lists.

        Most users should use :meth:`from_config` instead.

        Args:
            phase_modules: Mapping from phase name to ordered list of module
                instances.
            name_map: Optional mapping from module name to instance.  If not
                provided it is built from ``module.config.get("_name")``.
        """
        self._phase_modules: dict[str, list[BaseModule]] = {phase: list(modules) for phase, modules in phase_modules.items()}
        all_modules = self._all_unique_modules()

        if name_map is None:
            name_map = {module.config.get("_name", type(module).__name__): module for module in all_modules}
        self._name_map: dict[str, BaseModule] = dict(name_map)
        self.container = ModuleContainer()

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        registry: dict[str, type[BaseModule]] | None = None,
    ) -> ModuleManager:
        """Build a manager from a phase-based configuration dictionary.

        Args:
            config: Configuration dictionary containing ``modules`` plus one
                sub-dict per module.
            registry: Mapping from module names to module classes.  If ``None``,
                the default global registry is used.

        Returns:
            A :class:`ModuleManager` with instantiated modules in the order
            specified by each phase list under ``config["modules"]``.
        """
        if registry is None:
            registry = DEFAULT_REGISTRY

        modules_entry = config.get("modules", {})
        if not isinstance(modules_entry, dict):
            raise ValueError(f"'modules' must be a dict of phase lists, got {type(modules_entry).__name__}.")

        phase_modules: dict[str, list[BaseModule]] = {}
        name_map: dict[str, BaseModule] = {}

        for phase in cls._PHASE_NAMES:
            phase_list = modules_entry.get(phase, [])
            if phase_list is None:
                phase_list = []
            if isinstance(phase_list, dict) and not phase_list:
                phase_list = []
            if not isinstance(phase_list, list):
                raise ValueError(f"'{phase}' must be a list, got {type(phase_list).__name__}.")

            phase_instances: list[BaseModule] = []
            for entry in phase_list:
                name = cls._extract_module_name(entry)
                if name not in name_map:
                    if name not in registry:
                        available = ", ".join(sorted(registry.keys()))
                        raise KeyError(f"Module '{name}' is not registered. Available modules: {available}")

                    module_cls = registry[name]
                    module_config = config.get(name, {})
                    if not isinstance(module_config, dict):
                        raise ValueError(f"Config for module '{name}' must be a dict, got {type(module_config).__name__}.")
                    module_config = dict(module_config)
                    module_config.setdefault("_name", name)

                    instance = module_cls(module_config)
                    name_map[name] = instance

                phase_instances.append(name_map[name])

            phase_modules[phase] = phase_instances

        return cls(phase_modules, name_map)

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        registry: dict[str, type[BaseModule]] | None = None,
        context: dict[str, Any] | None = None,
    ) -> ModuleManager:
        """Build a manager from a YAML configuration file.

        Args:
            path: Path to the YAML file.
            registry: Optional registry; uses :data:`DEFAULT_REGISTRY` if
                ``None``.
            context: Optional runtime variables exposed to OmegaConf
                interpolations and expressions.

        Returns:
            A :class:`ModuleManager` configured by the YAML file.
        """
        config = load_yaml_with_context(path, context=context)
        return cls.from_config(config, registry=registry)

    @staticmethod
    def _extract_module_name(entry: Any) -> str:
        """Return the module name from a string or a {'name': ...} dict."""
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict):
            if "name" not in entry:
                raise ValueError(f"Module entry dict must contain a 'name' key, got {entry}.")
            return str(entry["name"])
        raise ValueError(f"Module entry must be a string or a dict, got {type(entry).__name__}: {entry}")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get(self, name: str) -> BaseModule:
        """Return a module by its registered/YAML name."""
        if name not in self._name_map:
            available = ", ".join(sorted(self._name_map.keys()))
            raise KeyError(f"Module '{name}' is not loaded. Loaded modules: {available}")
        return self._name_map[name]

    def __getitem__(self, name: str) -> BaseModule:
        return self.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._name_map

    def __iter__(self):
        return iter(self._all_unique_modules())

    def __len__(self) -> int:
        return len(self._all_unique_modules())

    def _all_unique_modules(self) -> list[BaseModule]:
        """Return every module once, in first-appearance order across phases."""
        seen: set[int] = set()
        all_modules: list[BaseModule] = []
        for phase in self._PHASE_NAMES:
            for module in self._phase_modules.get(phase, []):
                module_id = id(module)
                if module_id not in seen:
                    seen.add(module_id)
                    all_modules.append(module)
        return all_modules

    # ------------------------------------------------------------------
    # Lifecycle dispatch
    # ------------------------------------------------------------------
    def finalize(self) -> None:
        """Call ``finalize`` on every module in order."""
        self._call_hook("finalize")

    def post_finalize(self) -> None:
        """Call ``post_finalize`` on every module in order.

        Use this for one-shot construction that must happen after all modules
        have populated the shared container (e.g. finalizing the environment
        definition and creating the environment group).
        """
        self._call_hook("post_finalize")

    def pre_physics_step(self) -> None:
        """Call ``step`` on every pre-physics module in order."""
        self._call_step("pre_physics_step_modules")

    def pre_gym_step(self) -> None:
        """Call ``step`` on every pre-gym module in order."""
        self._call_step("pre_gym_step_modules")

    def post_gym_step(self) -> None:
        """Call ``step`` on every post-gym module in order."""
        self._call_step("post_gym_step_modules")

    def post_physics_step(self) -> None:
        """Call ``step`` on every post-physics module in order."""
        self._call_step("post_physics_step_modules")

    def reset(self) -> None:
        """Call ``reset`` on every module in order."""
        self._call_hook("reset")

    def _call_hook(self, hook: str) -> None:
        """Dispatch a hook to all unique modules that implement it."""
        for module in self._all_unique_modules():
            method = getattr(module, hook, None)
            if method is not None and callable(method):
                method(self.container)

    def _call_step(self, phase: str) -> None:
        """Dispatch ``step`` to all modules in the given phase."""
        for module in self._phase_modules.get(phase, []):
            module.step(self.container)

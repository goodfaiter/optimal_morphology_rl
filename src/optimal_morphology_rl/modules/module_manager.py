"""Module manager that loads modules from a config and dispatches lifecycle hooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, TypeVar

import yaml

from optimal_morphology_rl.modules.base_module import BaseModule


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
            raise ValueError(
                f"Module '{registry_name}' is already registered "
                f"({DEFAULT_REGISTRY[registry_name].__name__})."
            )
        DEFAULT_REGISTRY[registry_name] = cls
        return cls

    return decorator


class ModuleContainer:
    """Shared data store attached to :class:`ModuleManager`.

    Modules read and write simulation state here during their lifecycle hooks
    so that downstream modules can access it without hard-coded references.

    Access uses attribute syntax::

        container.gym = gym
        gym = container.gym
    """

    def __init__(self):
        self._data: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._data[name] = value

    def __contains__(self, name: str) -> bool:
        return name in self._data

    def get(self, name: str, default: Any = None) -> Any:
        return self._data.get(name, default)


class ModuleManager:
    """Owns an ordered list of modules and dispatches lifecycle hooks.

    Modules are instantiated from a config dict.  The special key
    ``config["modules"]`` must contain an ordered list of module names.  Each
    name is looked up in a registry and receives the sub-dict at
    ``config[name]``.

    Example YAML::

        modules:
          - create_rigid_vsim_envs
          - robot
          - object_generator

        create_rigid_vsim_envs:
          num_envs: 4096
          device: cuda:0

        robot:
          vsim_path: /path/to/hand.vsim

        object_generator:
          reward_object: drawer
          scene_objects:
            - table
    """

    def __init__(
        self,
        modules: list[BaseModule],
        name_map: dict[str, BaseModule] | None = None,
    ):
        """Create a manager from pre-built module instances.

        Most users should use :meth:`from_config` instead.

        Args:
            modules: Ordered list of module instances.
            name_map: Optional mapping from module name to instance.  If not
                provided it is built from ``module.config.get("_name")``.
        """
        self._modules: list[BaseModule] = list(modules)
        if name_map is None:
            name_map = {
                module.config.get("_name", type(module).__name__): module
                for module in modules
            }
        self._name_map: dict[str, BaseModule] = dict(name_map)
        self.container = ModuleContainer()

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        registry: dict[str, type[BaseModule]] | None = None,
    ) -> ModuleManager:
        """Build a manager from a configuration dictionary.

        Args:
            config: Configuration dictionary containing ``modules`` plus one
                sub-dict per module.
            registry: Mapping from module names to module classes.  If ``None``,
                the default global registry is used.

        Returns:
            A :class:`ModuleManager` with instantiated modules in the order
            specified by ``config["modules"]``.
        """
        if registry is None:
            registry = DEFAULT_REGISTRY

        module_names = config.get("modules", [])
        if not isinstance(module_names, list):
            raise ValueError(
                f"'modules' must be a list, got {type(module_names).__name__}."
            )

        instances: list[BaseModule] = []
        name_map: dict[str, BaseModule] = {}

        for entry in module_names:
            name = cls._extract_module_name(entry)
            if name not in registry:
                available = ", ".join(sorted(registry.keys()))
                raise KeyError(
                    f"Module '{name}' is not registered. Available modules: {available}"
                )

            module_cls = registry[name]
            module_config = config.get(name, {})
            if not isinstance(module_config, dict):
                raise ValueError(
                    f"Config for module '{name}' must be a dict, got "
                    f"{type(module_config).__name__}."
                )
            module_config = dict(module_config)
            module_config.setdefault("_name", name)

            instance = module_cls(module_config)
            instances.append(instance)
            name_map[name] = instance

        return cls(instances, name_map)

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        registry: dict[str, type[BaseModule]] | None = None,
    ) -> ModuleManager:
        """Build a manager from a YAML configuration file.

        Args:
            path: Path to the YAML file.
            registry: Optional registry; uses :data:`DEFAULT_REGISTRY` if
                ``None``.

        Returns:
            A :class:`ModuleManager` configured by the YAML file.
        """
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        if config is None:
            config = {}
        return cls.from_config(config, registry=registry)

    @staticmethod
    def _extract_module_name(entry: Any) -> str:
        """Return the module name from a string or a {'name': ...} dict."""
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict):
            if "name" not in entry:
                raise ValueError(
                    f"Module entry dict must contain a 'name' key, got {entry}."
                )
            return str(entry["name"])
        raise ValueError(
            f"Module entry must be a string or a dict, got {type(entry).__name__}: {entry}"
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get(self, name: str) -> BaseModule:
        """Return a module by its registered/YAML name."""
        if name not in self._name_map:
            available = ", ".join(sorted(self._name_map.keys()))
            raise KeyError(
                f"Module '{name}' is not loaded. Loaded modules: {available}"
            )
        return self._name_map[name]

    def __getitem__(self, name: str) -> BaseModule:
        return self.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._name_map

    def __iter__(self):
        return iter(self._modules)

    def __len__(self) -> int:
        return len(self._modules)

    # ------------------------------------------------------------------
    # Lifecycle dispatch
    # ------------------------------------------------------------------
    def finalize(self, env: Any) -> None:
        """Call ``finalize`` on every module in order."""
        self._call_hook("finalize", env)

    def post_finalize(self, env: Any) -> None:
        """Call ``post_finalize`` on every module in order.

        Use this for one-shot construction that must happen after all modules
        have populated the shared container (e.g. finalizing the environment
        definition and creating the environment group).
        """
        self._call_hook("post_finalize", env)

    def pre_physics_step(self, env: Any) -> None:
        """Call ``pre_physics_step`` on every module in order."""
        self._call_hook("pre_physics_step", env)

    def pre_gym_step(self, env: Any) -> None:
        """Call ``pre_gym_step`` on every module in order."""
        self._call_hook("pre_gym_step", env)

    def post_gym_step(self, env: Any) -> None:
        """Call ``post_gym_step`` on every module in order."""
        self._call_hook("post_gym_step", env)

    def post_physics_step(self, env: Any) -> None:
        """Call ``post_physics_step`` on every module in order."""
        self._call_hook("post_physics_step", env)

    def reset(self, env: Any) -> None:
        """Call ``reset`` on every module in order."""
        self._call_hook("reset", env)

    def _call_hook(self, hook: str, env: Any) -> None:
        """Dispatch a hook to all modules that implement it."""
        for module in self._modules:
            method = getattr(module, hook, None)
            if method is not None and callable(method):
                method(env)

"""Shared container for environment modules."""

from __future__ import annotations

from typing import Any


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

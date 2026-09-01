import unittest
from typing import Any

from optimal_morphology_rl.modules import (
    BaseModule,
    ModuleContainer,
    ModuleManager,
    register_module,
)


@register_module("alpha")
class AlphaModule(BaseModule):
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.calls: list[str] = []

    def finalize(self, container: ModuleContainer) -> None:
        self.calls.append("finalize")
        container.alpha_finalized = True

    def post_finalize(self, container: ModuleContainer) -> None:
        self.calls.append("post_finalize")
        container.alpha_post_finalized = True

    def step(self, container: ModuleContainer) -> None:
        self.calls.append("step")


@register_module("beta")
class BetaModule(BaseModule):
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.calls: list[str] = []

    def finalize(self, container: ModuleContainer) -> None:
        self.calls.append("finalize")
        container.beta_finalized = True

    def step(self, container: ModuleContainer) -> None:
        self.calls.append("step")

    def reset(self, container: ModuleContainer) -> None:
        self.calls.append("reset")


class TestModuleManager(unittest.TestCase):
    def test_basic_load_and_dispatch(self):
        config = {
            "modules": {
                "init_modules": ["alpha", "beta"],
                "post_physics_step_modules": ["alpha", "beta"],
            },
            "alpha": {"foo": 1},
            "beta": {"bar": 2},
        }

        manager = ModuleManager.from_config(config)

        self.assertEqual(len(manager), 2)
        self.assertIn("alpha", manager)
        self.assertIn("beta", manager)

        alpha = manager["alpha"]
        beta = manager["beta"]
        self.assertIsInstance(alpha, AlphaModule)
        self.assertIsInstance(beta, BetaModule)
        self.assertEqual(alpha.config["foo"], 1)
        self.assertEqual(beta.config["bar"], 2)

        self.assertIsInstance(manager.container, ModuleContainer)

        manager.finalize()
        self.assertEqual(alpha.calls, ["finalize"])
        self.assertEqual(beta.calls, ["finalize"])
        self.assertTrue(manager.container.alpha_finalized)
        self.assertTrue(manager.container.beta_finalized)

        manager.post_finalize()
        self.assertEqual(alpha.calls, ["finalize", "post_finalize"])
        self.assertEqual(beta.calls, ["finalize"])
        self.assertTrue(manager.container.alpha_post_finalized)

        manager.post_physics_step()
        self.assertEqual(alpha.calls, ["finalize", "post_finalize", "step"])
        self.assertEqual(beta.calls, ["finalize", "step"])

        manager.reset()
        self.assertEqual(beta.calls, ["finalize", "step", "reset"])

    def test_order_matches_yaml(self):
        config = {
            "modules": {
                "init_modules": ["beta", "alpha"],
            },
        }
        manager = ModuleManager.from_config(config)
        names = [m.config["_name"] for m in manager]
        self.assertEqual(names, ["beta", "alpha"])

    def test_missing_module_raises(self):
        config = {"modules": {"init_modules": ["nonexistent"]}}
        with self.assertRaises(KeyError) as cm:
            ModuleManager.from_config(config)
        self.assertIn("nonexistent", str(cm.exception))

    def test_get_missing_module_raises(self):
        manager = ModuleManager.from_config({"modules": {}})
        with self.assertRaises(KeyError) as cm:
            manager.get("alpha")
        self.assertIn("alpha", str(cm.exception))

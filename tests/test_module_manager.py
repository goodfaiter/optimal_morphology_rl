import unittest
from typing import Any

from optimal_morphology_rl.modules import BaseModule, ModuleContainer, ModuleManager, register_module


@register_module("alpha")
class AlphaModule(BaseModule):
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.calls: list[str] = []
        self.env_seen: Any = None

    def finalize(self, env: Any) -> None:
        self.calls.append("finalize")
        self.env_seen = env
        env.module_manager.container.alpha_finalized = True

    def post_finalize(self, env: Any) -> None:
        self.calls.append("post_finalize")
        env.module_manager.container.alpha_post_finalized = True

    def pre_physics_step(self, env: Any) -> None:
        self.calls.append("pre_physics_step")

    def post_physics_step(self, env: Any) -> None:
        self.calls.append("post_physics_step")


@register_module("beta")
class BetaModule(BaseModule):
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.calls: list[str] = []

    def finalize(self, env: Any) -> None:
        self.calls.append("finalize")
        env.module_manager.container.beta_finalized = True

    def pre_physics_step(self, env: Any) -> None:
        self.calls.append("pre_physics_step")

    def post_physics_step(self, env: Any) -> None:
        self.calls.append("post_physics_step")

    def reset(self, env: Any) -> None:
        self.calls.append("reset")


class DummyEnv:
    def __init__(self, manager: ModuleManager):
        self.module_manager = manager


class TestModuleManager(unittest.TestCase):
    def test_basic_load_and_dispatch(self):
        config = {
            "modules": ["alpha", "beta"],
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

        env = DummyEnv(manager)
        manager.finalize(env)
        self.assertEqual(alpha.calls, ["finalize"])
        self.assertEqual(alpha.env_seen, env)
        self.assertEqual(beta.calls, ["finalize"])
        self.assertTrue(manager.container.alpha_finalized)
        self.assertTrue(manager.container.beta_finalized)

        manager.post_finalize(env)
        self.assertEqual(alpha.calls, ["finalize", "post_finalize"])
        self.assertEqual(beta.calls, ["finalize"])
        self.assertTrue(manager.container.alpha_post_finalized)

        manager.pre_physics_step(env)
        self.assertEqual(alpha.calls, ["finalize", "post_finalize", "pre_physics_step"])
        self.assertEqual(beta.calls, ["finalize", "pre_physics_step"])

        manager.post_physics_step(env)
        self.assertEqual(alpha.calls, ["finalize", "post_finalize", "pre_physics_step", "post_physics_step"])
        self.assertEqual(beta.calls, ["finalize", "pre_physics_step", "post_physics_step"])

        manager.reset(env)
        self.assertEqual(beta.calls, ["finalize", "pre_physics_step", "post_physics_step", "reset"])

    def test_order_matches_yaml(self):
        config = {
            "modules": ["beta", "alpha"],
        }
        manager = ModuleManager.from_config(config)
        names = [m.config["_name"] for m in manager]
        self.assertEqual(names, ["beta", "alpha"])

    def test_missing_module_raises(self):
        config = {"modules": ["nonexistent"]}
        with self.assertRaises(KeyError) as cm:
            ModuleManager.from_config(config)
        self.assertIn("nonexistent", str(cm.exception))

    def test_get_missing_module_raises(self):
        manager = ModuleManager.from_config({"modules": []})
        with self.assertRaises(KeyError) as cm:
            manager.get("alpha")
        self.assertIn("alpha", str(cm.exception))

    def test_invalid_modules_type(self):
        config = {"modules": "alpha"}
        with self.assertRaises(ValueError) as cm:
            ModuleManager.from_config(config)
        self.assertIn("list", str(cm.exception))

    def test_dict_entry_name(self):
        config = {"modules": [{"name": "alpha"}]}
        manager = ModuleManager.from_config(config)
        self.assertIn("alpha", manager)


if __name__ == "__main__":
    unittest.main()

"""Integration test for the modular rigid-body environment creation."""

import os

import pytest
import torch
import vlearn as v

from optimal_morphology_rl.modules import ModuleManager


VSIM_PATH = "/workspace/data/vsim/claw_3_tendon.vsim"


class DummyEnv:
    def __init__(self, manager: ModuleManager):
        self.module_manager = manager
        self.reset_buf = torch.zeros(1, dtype=torch.bool, device="cuda:0")


@pytest.fixture(scope="module")
def _manager_session():
    """Build and finalize the module manager once for this test module."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    if not os.path.isfile(VSIM_PATH):
        pytest.fail(f"Hand VSIM not found: {VSIM_PATH}")

    config = {
        "modules": ["create_rigid_vsim_envs", "robot", "object_generator"],
        "create_rigid_vsim_envs": {
            "num_envs": 1,
            "device": "cuda:0",
            "rendering": False,
            "with_window": False,
            "enable_scene_query": True,
            "max_episode_length": 10,
        },
        "robot": {
            "vsim_path": VSIM_PATH,
            "fixed_hand": False,
            "use_tendon": True,
        },
        "object_generator": {
            "reward_object": "drawer",
            "scene_objects": ["table"],
        },
    }

    manager = ModuleManager.from_config(config)
    env = DummyEnv(manager)
    manager.finalize(env)
    manager.post_finalize(env)

    yield manager

    # Break references to the singleton Gym so the next test module can
    # create its own.
    container = manager.container
    container.gym = None
    container.env_def = None
    container.env_group = None
    container.env_sets = None
    container.env_set = None
    container.env_set_handle = None
    container.env_set_handles = None
    container.robot = None
    container.objects = None
    container.reward_object = None
    v.delete_gym()


@pytest.fixture
def manager(_manager_session):
    return _manager_session


class TestCreateRigidVsimEnvs:
    def test_container_has_gym_and_env_group(self, manager):
        container = manager.container
        assert container.gym is not None
        assert container.env_def is not None
        assert container.env_group is not None
        assert container.num_envs == 1
        assert container.total_num_envs == 1

    def test_robot_loaded(self, manager):
        container = manager.container
        assert container.robot is not None
        assert container.robot.arti_handle is not None

    def test_objects_loaded(self, manager):
        container = manager.container
        assert container.objects is not None
        assert isinstance(container.objects, dict)
        assert container.reward_object is not None
        assert container.reward_object.name == "drawer"
        assert "table" in container.objects
        assert container.objects["table"].name == "table"

    def test_object_buffers_allocated(self, manager):
        """Object generator allocated state buffers and created GPU commands."""
        container = manager.container
        reward_object = container.reward_object
        assert reward_object.get_trans_object_to_world_buf is not None
        assert reward_object.get_trans_object_to_world_buf.shape[0] == container.total_num_envs
        assert reward_object.gpu_get_object_kin_cmd_array is not None

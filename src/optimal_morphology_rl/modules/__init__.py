"""Environment modules and module manager."""

# Import modules so their @register_module decorators execute.
from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.create_rigid_vsim_envs import CreateRigidVsimEnvs
from optimal_morphology_rl.modules.module_manager import (
    DEFAULT_REGISTRY,
    ModuleContainer,
    ModuleManager,
    register_module,
)
from optimal_morphology_rl.modules.object_generator_module import ObjectGeneratorModule
from optimal_morphology_rl.modules.robot_module import RobotModule

__all__ = [
    "BaseModule",
    "CreateRigidVsimEnvs",
    "ModuleContainer",
    "ModuleManager",
    "ObjectGeneratorModule",
    "RobotModule",
    "register_module",
    "DEFAULT_REGISTRY",
]

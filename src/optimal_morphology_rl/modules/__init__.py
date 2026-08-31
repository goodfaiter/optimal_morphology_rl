"""Environment modules and module manager."""

# Import modules so their @register_module decorators execute.
from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.camera_recorder_module import CameraRecorderModule
from optimal_morphology_rl.modules.contacts_module import ContactsModule
from optimal_morphology_rl.modules.create_rigid_vsim_envs import CreateRigidVsimEnvs
from optimal_morphology_rl.modules.external_force_module import ExternalForceModule
from optimal_morphology_rl.modules.force_sensors_module import ForceSensorsModule
from optimal_morphology_rl.modules.kinematic_sensor_module import KinematicSensorModule
from optimal_morphology_rl.modules.module_manager import (
    DEFAULT_REGISTRY,
    ModuleContainer,
    ModuleManager,
    register_module,
)
from optimal_morphology_rl.modules.object_generator_module import ObjectGeneratorModule

# Observation sub-modules and manager.
from optimal_morphology_rl.modules.observations import (
    OBSERVATION_REGISTRY,
    GoalStateObservation,
    ObjectStateObservation,
    ObservationBaseModule,
    ObservationManagerModule,
    RobotStateObservation,
    register_observation,
)
from optimal_morphology_rl.modules.rewards import (
    REWARD_REGISTRY,
    RewardBaseModule,
    RewardManagerModule,
    register_reward,
)
from optimal_morphology_rl.modules.robot_control_module import RobotControlModule
from optimal_morphology_rl.modules.robot_module import RobotModule
from optimal_morphology_rl.modules.terminations import (
    TERMINATION_REGISTRY,
    TerminationBaseModule,
    TerminationManagerModule,
    register_termination,
)

# Import environment-specific reward/termination modules so their decorators
# register even when only the module registry is imported.
from optimal_morphology_rl.envs.hand_cube import rewards as _hand_cube_rewards
from optimal_morphology_rl.envs.hand_cube import terminations as _hand_cube_terminations

__all__ = [
    "BaseModule",
    "CameraRecorderModule",
    "ContactsModule",
    "CreateRigidVsimEnvs",
    "ExternalForceModule",
    "ForceSensorsModule",
    "KinematicSensorModule",
    "ModuleContainer",
    "ModuleManager",
    "ObjectGeneratorModule",
    "ObservationBaseModule",
    "ObservationManagerModule",
    "RewardBaseModule",
    "RewardManagerModule",
    "RobotControlModule",
    "RobotModule",
    "TerminationBaseModule",
    "TerminationManagerModule",
    "register_module",
    "register_observation",
    "register_reward",
    "register_termination",
    "DEFAULT_REGISTRY",
    "OBSERVATION_REGISTRY",
    "REWARD_REGISTRY",
    "TERMINATION_REGISTRY",
    "RobotStateObservation",
    "ObjectStateObservation",
    "GoalStateObservation",
]

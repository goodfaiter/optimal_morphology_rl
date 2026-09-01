"""Environment modules and module manager."""

# Import modules so their @register_module decorators execute.
from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.camera_recorder_module import CameraRecorderModule
from optimal_morphology_rl.modules.create_contacts_module import CreateContactsModule
from optimal_morphology_rl.modules.update_contacts_module import UpdateContactsModule
from optimal_morphology_rl.modules.create_rigid_vsim_envs import CreateRigidVsimEnvs
from optimal_morphology_rl.modules.external_force_module import ExternalForceModule
from optimal_morphology_rl.modules.create_kinematic_sensor_module import CreateKinematicSensorModule
from optimal_morphology_rl.modules.update_kinematic_sensor_module import UpdateKinematicSensorModule
from optimal_morphology_rl.modules.create_force_sensor_module import CreateForceSensorModule
from optimal_morphology_rl.modules.update_force_sensors_module import UpdateForceSensorsModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import (
    DEFAULT_REGISTRY,
    ModuleManager,
    register_module,
)
from optimal_morphology_rl.modules.color_articulation_links_module import (
    ColorArticulationLinksModule,
)
from optimal_morphology_rl.modules.create_objects_module import CreateObjectsModule
from optimal_morphology_rl.modules.object_control_module import ObjectControlModule
from optimal_morphology_rl.modules.process_actions_module import ProcessActionsModule
from optimal_morphology_rl.modules.update_objects_module import UpdateObjectsModule

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
from optimal_morphology_rl.modules.create_robot_module import RobotModule
from optimal_morphology_rl.modules.update_robot_module import UpdateRobotModule
from optimal_morphology_rl.modules.terminations import (
    TERMINATION_REGISTRY,
    TerminationBaseModule,
    TerminationManagerModule,
    register_termination,
)
from optimal_morphology_rl.modules.visualization.goal_visualization_module import (
    GoalVisualizationModule,
)
from optimal_morphology_rl.modules.visualization.render_module import RenderModule

# Import environment-specific reward/termination modules so their decorators
# register even when only the module registry is imported.
from optimal_morphology_rl.envs.hand_cube import rewards as _hand_cube_rewards
from optimal_morphology_rl.envs.hand_cube import terminations as _hand_cube_terminations

__all__ = [
    "ColorArticulationLinksModule",
    "BaseModule",
    "CameraRecorderModule",
    "CreateContactsModule",
    "UpdateContactsModule",
    "CreateRigidVsimEnvs",
    "ExternalForceModule",
    "GoalVisualizationModule",
    "CreateKinematicSensorModule",
    "UpdateKinematicSensorModule",
    "CreateForceSensorModule",
    "UpdateForceSensorsModule",
    "ModuleContainer",
    "ModuleManager",
    "CreateObjectsModule",
    "ObjectControlModule",
    "UpdateObjectsModule",
    "ObservationBaseModule",
    "ObservationManagerModule",
    "ProcessActionsModule",
    "RenderModule",
    "RewardBaseModule",
    "RewardManagerModule",
    "RobotControlModule",
    "RobotModule",
    "UpdateRobotModule",
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

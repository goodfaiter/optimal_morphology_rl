"""Observation sub-modules and manager."""

# Import observation sub-modules so their @register_observation decorators run.
from optimal_morphology_rl.modules.observations.goal_state_observation import (
    GoalStateObservation,
)
from optimal_morphology_rl.modules.observations.object_state_observation import (
    ObjectStateObservation,
)
from optimal_morphology_rl.modules.observations.observation_base_module import (
    ObservationBaseModule,
)
from optimal_morphology_rl.modules.observations.observation_manager_module import (
    OBSERVATION_REGISTRY,
    ObservationManagerModule,
    register_observation,
)
from optimal_morphology_rl.modules.observations.robot_state_observation import (
    RobotStateObservation,
)

__all__ = [
    "ObservationBaseModule",
    "ObservationManagerModule",
    "register_observation",
    "OBSERVATION_REGISTRY",
    "RobotStateObservation",
    "ObjectStateObservation",
    "GoalStateObservation",
]

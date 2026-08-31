"""Reward sub-modules and manager."""

# Import generic reward modules so their @register_reward decorators run.
from optimal_morphology_rl.modules.rewards.action_penalty_reward import (
    ActionPenaltyReward,
)
from optimal_morphology_rl.modules.rewards.action_smoothness_reward import (
    ActionSmoothnessReward,
)
from optimal_morphology_rl.modules.rewards.fingertip_contact_reward import (
    FingertipContactReward,
)
from optimal_morphology_rl.modules.rewards.goal_orientation_reward import (
    GoalOrientationReward,
)
from optimal_morphology_rl.modules.rewards.goal_position_reward import (
    GoalPositionReward,
)
from optimal_morphology_rl.modules.rewards.hand_to_object_distance_reward import (
    HandToObjectDistanceReward,
)
from optimal_morphology_rl.modules.rewards.reward_base_module import RewardBaseModule
from optimal_morphology_rl.modules.rewards.reward_manager_module import (
    REWARD_REGISTRY,
    RewardManagerModule,
    register_reward,
)

__all__ = [
    "RewardBaseModule",
    "RewardManagerModule",
    "register_reward",
    "REWARD_REGISTRY",
    "GoalPositionReward",
    "GoalOrientationReward",
    "HandToObjectDistanceReward",
    "FingertipContactReward",
    "ActionPenaltyReward",
    "ActionSmoothnessReward",
]

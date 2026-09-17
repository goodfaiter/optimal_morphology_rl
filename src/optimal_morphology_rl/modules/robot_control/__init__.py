"""Robot-control compute modules: per-hand-type control computation and shared helpers."""

from optimal_morphology_rl.modules.robot_control.helpers import build_active_dof_mask, get_num_actions
from optimal_morphology_rl.modules.robot_control.robot_control_floating_hand_module import RobotControlFloatingHandModule
from optimal_morphology_rl.modules.robot_control.robot_control_motors_module import RobotControlMotorsModule
from optimal_morphology_rl.modules.robot_control.robot_control_tendons_module import RobotControlTendonsModule

__all__ = [
    "RobotControlFloatingHandModule",
    "RobotControlMotorsModule",
    "RobotControlTendonsModule",
    "build_active_dof_mask",
    "get_num_actions",
]

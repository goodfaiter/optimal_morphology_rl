"""Apply modules: apply the computed robot controls via vlearn gym commands."""

from optimal_morphology_rl.modules.apply_control.apply_motor_forces_module import ApplyMotorForcesModule
from optimal_morphology_rl.modules.apply_control.apply_root_forces_module import ApplyRootForcesModule
from optimal_morphology_rl.modules.apply_control.apply_root_velocity_module import ApplyRootVelocityModule
from optimal_morphology_rl.modules.apply_control.apply_tendon_forces_module import ApplyTendonForcesModule

__all__ = [
    "ApplyMotorForcesModule",
    "ApplyRootForcesModule",
    "ApplyRootVelocityModule",
    "ApplyTendonForcesModule",
]

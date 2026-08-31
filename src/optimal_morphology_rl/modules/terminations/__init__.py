"""Termination sub-modules and manager."""

# Import generic termination modules so their @register_termination decorators run.
from optimal_morphology_rl.modules.terminations.bounds_termination import (
    BoundsTermination,
)
from optimal_morphology_rl.modules.terminations.drop_termination import DropTermination
from optimal_morphology_rl.modules.terminations.termination_base_module import (
    TerminationBaseModule,
)
from optimal_morphology_rl.modules.terminations.termination_manager_module import (
    TERMINATION_REGISTRY,
    TerminationManagerModule,
    register_termination,
)

__all__ = [
    "TerminationBaseModule",
    "TerminationManagerModule",
    "register_termination",
    "TERMINATION_REGISTRY",
    "DropTermination",
    "BoundsTermination",
]

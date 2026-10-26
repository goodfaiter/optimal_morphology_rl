"""curriculum_range: randomization ranges shrink toward the nominal value with the training curriculum."""

import torch

from optimal_morphology_rl.modules.base_module import curriculum_range


def test_curriculum_range_shrinks_to_the_nominal() -> None:
    assert curriculum_range(0.8, 1.2, 0.0) == (1.0, 1.0)
    assert curriculum_range(0.8, 1.2, 1.0) == (0.8, 1.2)
    low, high = curriculum_range(0.0, 0.1, 0.5, nominal=0.0)
    assert (low, high) == (0.0, 0.05)


def test_curriculum_range_per_element_nominal() -> None:
    nominal = torch.tensor([0.038, 0.036])
    low, high = curriculum_range(0.032, 0.040, 0.0, nominal=nominal)
    assert torch.equal(low, nominal) and torch.equal(high, nominal)

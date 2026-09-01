"""TorchScript helpers for reward computations."""

from __future__ import annotations

import torch
from vlearn.torch_utils.torch_jit_utils import quat_rotate


@torch.jit.script
def _quaternion_to_6d_jit(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (x, y, z, w) to 6D rotation representation.

    The 6D representation is the first two columns of the rotation matrix.
    """
    num_envs = q.shape[0]
    device = q.device

    e1 = torch.tensor([1.0, 0.0, 0.0], device=device).unsqueeze(0).expand(num_envs, 3)
    e2 = torch.tensor([0.0, 1.0, 0.0], device=device).unsqueeze(0).expand(num_envs, 3)

    col1 = quat_rotate(q, e1)
    col2 = quat_rotate(q, e2)
    return torch.cat([col1, col2], dim=-1)


@torch.jit.script
def _goal_position_reward_jit(
    object_pos: torch.Tensor,
    goal_pos: torch.Tensor,
    normalize: float,
    scale: float,
) -> torch.Tensor:
    """Exponential reward based on object-to-goal distance."""
    dist = torch.norm(object_pos - goal_pos, dim=-1)
    return scale * torch.exp(-((dist / normalize) ** 2))


@torch.jit.script
def _goal_orientation_reward_jit(
    object_quat: torch.Tensor,
    goal_quat: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Dot-product reward between object and goal 6D rotation representations."""
    object_6d = _quaternion_to_6d_jit(object_quat)
    goal_6d = _quaternion_to_6d_jit(goal_quat)
    alignment = torch.sum(goal_6d * object_6d, dim=-1) / 2.0
    return scale * alignment


@torch.jit.script
def _hand_to_object_distance_reward_jit(
    link_positions: torch.Tensor,
    object_pos: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Exponential reward based on average distal-link-to-object distance."""
    # link_positions: (N, L, 3), object_pos: (N, 3)
    link_dists = torch.norm(link_positions - object_pos.unsqueeze(1), dim=-1)
    avg_dist = link_dists.mean(dim=-1)
    dist_clipped = torch.clamp(avg_dist, min=0.01)
    dist_rew = torch.exp(-((dist_clipped / 0.2) ** 2))
    return scale * dist_rew


@torch.jit.script
def _fingertip_contact_reward_jit(
    forces: torch.Tensor,
    contact_mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Reward contact between monitored distal links and the reward object."""
    force_against_object = forces * contact_mask
    force_against_object = torch.clamp(force_against_object, 0.0, 1.0)
    contact_reward = torch.clamp(force_against_object.sum(dim=-1), 0.0, 3.0) / 3.0
    return scale * contact_reward


@torch.jit.script
def _action_penalty_reward_jit(actions: torch.Tensor, scale: float) -> torch.Tensor:
    """Negative reward proportional to squared action magnitude."""
    return -scale * torch.sum(actions * actions, dim=-1)


@torch.jit.script
def _action_smoothness_reward_jit(
    actions: torch.Tensor,
    last_actions: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Negative reward proportional to squared action delta."""
    delta = actions - last_actions
    return -scale * torch.sum(delta * delta, dim=-1)

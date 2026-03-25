import torch
from vlearn.torch_utils.torch_jit_utils import scale, quat_mul, quat_conjugate, v_rpy_from_quat, quat_rotate



def world_down_in_robot_frame_from_quat_robot_to_world(quat_robot_to_world: torch.Tensor) -> torch.Tensor:
    quat_world_to_robot = quat_conjugate(quat_robot_to_world)
    down_in_world = (
        torch.tensor([0, 0, -1], device=quat_world_to_robot.device, dtype=quat_world_to_robot.dtype)
        .unsqueeze(0)
        .expand(quat_world_to_robot.shape[0], -1)
    )
    return quat_rotate(quat_world_to_robot, down_in_world)


def palm_down_in_world_frame_from_quat_robot_to_world(quat_robot_to_world: torch.Tensor) -> torch.Tensor:
    palm_down_in_world = (
        torch.tensor([0, 1, 0], device=quat_robot_to_world.device, dtype=quat_robot_to_world.dtype)
        .unsqueeze(0)
        .expand(quat_robot_to_world.shape[0], -1)
    )
    return quat_rotate(quat_robot_to_world, palm_down_in_world)


def rotate_by_quat_A_to_B(quat_A_to_B: torch.Tensor, vec_A: torch.Tensor) -> torch.Tensor:
    return quat_rotate(quat_A_to_B, vec_A)


def pen_forward_in_world_frame_from_quat_pen_to_world(quat_pen_to_world: torch.Tensor) -> torch.Tensor:
    pen_forward_in_world = (
        torch.tensor([1, 0, 0], device=quat_pen_to_world.device, dtype=quat_pen_to_world.dtype)
        .unsqueeze(0)
        .expand(quat_pen_to_world.shape[0], -1)
    )
    return quat_rotate(quat_pen_to_world, pen_forward_in_world)
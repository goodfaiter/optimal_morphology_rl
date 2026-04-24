import numpy as np
import vlearn as v
import torch
from vlearn.torch_utils.torch_jit_utils import quaternion_to_matrix


def vec3_to_numpy(vec: v.Vec3) -> np.ndarray:
    return np.array([vec.x, vec.y, vec.z], dtype=np.float32)


def quat_to_numpy(quat: v.Quat) -> np.ndarray:
    return np.array([quat.x, quat.y, quat.z, quat.w], dtype=np.float32)


def numpy_to_vec3(arr: np.ndarray) -> v.Vec3:
    return v.Vec3(arr[0], arr[1], arr[2])


def numpy_to_quat(arr: np.ndarray) -> v.Quat:
    return v.Quat(arr[0], arr[1], arr[2], arr[3])


def random_uniform_quaternion(num_envs, device, dtype):
    q = torch.randn(num_envs, 4, device=device, dtype=dtype)  # 4 independent Gaussians
    return q / torch.norm(q, dim=1, keepdim=True)  # Normalize to unit length


def quaternion_to_6d(q):
    """
    Convert quaternion to 6D continuous representation.
    
    Args:
        q: Quaternion tensor of shape [..., 4] (x, y, z, w)
    
    Returns:
        6D vector of shape [..., 6] (first two columns of rotation matrix)
    """
    R = quaternion_to_matrix(q)
    # Extract first two columns and flatten
    return torch.cat([R[..., 0], R[..., 1]], dim=-1)  # Shape: [..., 6]
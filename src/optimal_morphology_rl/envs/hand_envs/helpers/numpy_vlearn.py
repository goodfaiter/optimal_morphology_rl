import numpy as np
import vlearn as v
import torch


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
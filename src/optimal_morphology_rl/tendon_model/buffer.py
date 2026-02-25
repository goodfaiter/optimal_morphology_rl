import torch

class TimeSeriesBuffer:
    def __init__(self, num_envs, max_size, feature_dim=None, device='cpu'):
        self.feature_dim = feature_dim
        self.max_size = max_size
        if feature_dim is None:
            self.buffer = torch.zeros((num_envs, max_size), device=device)
        else:
            self.buffer = torch.zeros((num_envs, max_size, feature_dim), device=device)

    def add(self, value: torch.Tensor):
        self.buffer = torch.roll(self.buffer, -1, dims=1)
        if self.feature_dim is None:
            self.buffer[:, -1] = value.squeeze(-1)
        else:
            self.buffer[:, -1, :] = value

    def reset(self, reset_mask: torch.Tensor):
        self.buffer[reset_mask] = 0.0
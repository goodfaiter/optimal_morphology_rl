import torch

class TimeSeriesBuffer:
    def __init__(self, num_envs, max_size, device='cpu'):
        self.buffer = torch.zeros((num_envs, max_size), device=device)
        self.max_size = max_size
    
    def add(self, value):
        self.buffer = torch.roll(self.buffer, -1, dims=1)
        self.buffer[:, -1] = value.squeeze()

    def reset(self, reset_mask: torch.Tensor):
        self.buffer[reset_mask, :] = 0.0
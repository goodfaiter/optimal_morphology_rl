import torch


class TimeSeriesBuffer:
    def __init__(self, num_envs, dim:int, max_size: int, stride: int, device='cpu'):
        self.buffer = torch.zeros((num_envs, max_size, dim), device=device)
        self.max_size = max_size
        self.stride = stride
    
    def add(self, value):
        self.buffer = torch.roll(self.buffer, -1, dims=1)
        self.buffer[:, -1, :] = value.squeeze()

    def get(self):
        indices = torch.arange(self.max_size - 1, -1, -self.stride, device=self.buffer.device)
        return self.buffer[:, indices, :]

    def reset(self, reset_mask: torch.Tensor):
        self.buffer[reset_mask, :, :] = 0.0

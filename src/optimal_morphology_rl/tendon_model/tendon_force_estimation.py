import torch
from .buffer import TimeSeriesBuffer


class TendonForceEstimation:
    def __init__(self, file_path, num_envs, device):
        self._device = device
        self._file_path = file_path
        self._num_envs = num_envs
        self._load_model()
        self._initialize_buffers()

    def _load_model(self):
        self._model = torch.jit.load(self._file_path, map_location=self._device)
        self._seq_length = int(self._model.history_size.item())
        self._input_cols = self._model.input_columns
        self._output_cols = self._model.output_columns
        self._input = torch.zeros((self._num_envs, self._seq_length, len(self._input_cols)), device=self._device)
        self._output = torch.zeros((self._num_envs, 1, len(self._output_cols)), device=self._device)

    def reset(self, reset_mask: torch.Tensor):
        self.desired_position_buffer.reset(reset_mask)
        self.measured_position_buffer.reset(reset_mask)
        self.measured_velocity_buffer.reset(reset_mask)
        self.tendon_force_buffer.reset(reset_mask)

    def _initialize_buffers(self):
        self.desired_position_buffer = TimeSeriesBuffer(num_envs=self._num_envs, max_size=self._seq_length, device=self._device)
        self.measured_position_buffer = TimeSeriesBuffer(num_envs=self._num_envs, max_size=self._seq_length, device=self._device)
        self.measured_velocity_buffer = TimeSeriesBuffer(num_envs=self._num_envs, max_size=self._seq_length, device=self._device)
        self.tendon_force_buffer = TimeSeriesBuffer(num_envs=self._num_envs, max_size=self._seq_length, device=self._device)

    def desired_position(self, data: torch.Tensor):
        self.desired_position_buffer.add(data)

    def measured_position(self, data: torch.Tensor):
        self.measured_position_buffer.add(data)

    def measured_velocity(self, data: torch.Tensor):
        self.measured_velocity_buffer.add(data)

    def forward(self):
        self._input[:, :, 0] = self.desired_position_buffer.buffer[:]
        self._input[:, :, 1] = self.measured_position_buffer.buffer[:]
        self._input[:, :, 2] = self.measured_velocity_buffer.buffer[:]

        self._output[:] = self._model(self._input)
        return self._output[:, -1, 0]

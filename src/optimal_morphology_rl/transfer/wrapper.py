import torch


class EncoderWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, input_mean: torch.Tensor, input_std: torch.Tensor):
        super().__init__()
        self.model = model
        self.add_module("model", model)
        self.register_buffer("input_mean", input_mean.detach().clone().requires_grad_(False))
        self.register_buffer("input_std", input_std.detach().clone().requires_grad_(False))
        self.eps = 1e-6

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.input_mean) / (self.input_std + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self._normalize(x))

    def freeze(self) -> None:
        self.eval()
        self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze model weights."""
        self.train()
        self.model.train()
        for param in self.parameters():
            param.requires_grad = True

    def trace_and_save(self, save_path: str) -> None:
        self.eval()
        traced_model = torch.jit.script(self)
        traced_model.save(save_path)


class DecoderWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, output_mean: torch.Tensor, output_std: torch.Tensor):
        super().__init__()
        self.model = model
        self.add_module("model", model)
        # Store normalization stats as buffers so they are part of the scripted module
        self.register_buffer("output_mean", output_mean.detach().clone().requires_grad_(False))
        self.register_buffer("output_std", output_std.detach().clone().requires_grad_(False))
        self.eps = 1e-6

    def _denormalize(self, z: torch.Tensor) -> torch.Tensor:
        return z * (self.output_std + self.eps) + self.output_mean

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self._denormalize(self.model(z))

    def freeze(self) -> None:
        self.eval()
        self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze model weights."""
        self.train()
        self.model.train()
        for param in self.parameters():
            param.requires_grad = True

    def trace_and_save(self, save_path: str) -> None:
        self.eval()
        traced_model = torch.jit.script(self)
        traced_model.save(save_path)


# Example usage
if __name__ == "__main__":
    from optimal_morphology_rl.transfer.encoder import LatentSpaceEncoder

    # Wrap individual models separately
    teacher_encoder = LatentSpaceEncoder(input_size=50, latent_size=32)

    wrapped_encoder = EncoderWrapper(teacher_encoder, torch.zeros(50), torch.ones(50))

    # Freeze for deployment
    wrapped_encoder.freeze()

    # Save each model independently
    wrapped_encoder.trace_and_save("src/optimal_morphology_rl/transfer/teacher_encoder.pt")

    # Load and use
    x = torch.randn(1, 50)
    z = wrapped_encoder(x)
    print(f"Input: {x.shape} → Latent: {z.shape}")

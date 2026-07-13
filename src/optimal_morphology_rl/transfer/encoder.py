import torch


class LatentSpaceEncoder(torch.nn.Module):
    def __init__(self, input_size, latent_size, layers=[128, 64], device='cpu'):
        super(LatentSpaceEncoder, self).__init__()

        self.input_size = input_size
        self.latent_size = latent_size

        self.sequential = torch.nn.Sequential()
        for i, layer in enumerate(layers):
            if i == 0:
                self.sequential.add_module(f"fc{i+1}", torch.nn.Linear(input_size, layer, device=device))
            else:
                self.sequential.add_module(f"fc{i+1}", torch.nn.Linear(layers[i-1], layer, device=device))
            self.sequential.add_module(f"tanh_{i+1}", torch.nn.Tanh())
        self.sequential.add_module(f"fc{len(layers)+1}", torch.nn.Linear(layers[-1], latent_size, device=device))
        self.sequential.add_module("batch_normalization", torch.nn.BatchNorm1d(latent_size, device=device))
        
    def forward(self, x):
        return self.sequential(x)


class LatentSpaceDecoder(torch.nn.Module):
    def __init__(self, latent_size, output_size, layers=[64, 128], device='cpu'):
        super(LatentSpaceDecoder, self).__init__()

        self.latent_size = latent_size
        self.output_size = output_size

        self.sequential = torch.nn.Sequential()
        for i, layer in enumerate(layers):
            if i == 0:
                self.sequential.add_module(f"tanh_0", torch.nn.Tanh())
                self.sequential.add_module(f"fc{i+1}", torch.nn.Linear(latent_size, layer, device=device))
            if i > 0:
                self.sequential.add_module(f"fc{i+1}", torch.nn.Linear(layers[i-1], layer, device=device))
            self.sequential.add_module(f"tanh_{i+1}", torch.nn.Tanh())
        self.sequential.add_module(f"fc{len(layers)+1}", torch.nn.Linear(layers[-1], output_size, device=device))

    def forward(self, z):
        return self.sequential(z)

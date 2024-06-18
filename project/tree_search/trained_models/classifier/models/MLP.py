import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 n_hidden: int = 1,
                 latent_dim: int = 32,
                 use_batchnorm: bool = False,
                 activation: str = 'PReLU') -> None:
        super(MLP, self).__init__()

        # Dictionary mapping string names to activation functions
        activations = {
            'ReLU': nn.ReLU,
            'PReLU': nn.PReLU,
            'LeakyReLU': nn.LeakyReLU,
            'Sigmoid': nn.Sigmoid,
            'Tanh': nn.Tanh,
            'ELU': nn.ELU,
            'GELU': nn.GELU,
            'Mish': nn.Mish,
            'SELU': nn.SELU
        }

        if activation not in activations:
            raise ValueError(f"Activation function '{activation}' is not supported.")

        layers = [nn.Linear(input_dim, latent_dim), activations[activation]()]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(latent_dim))

        for _ in range(n_hidden):
            layers += [nn.Linear(latent_dim, latent_dim), activations[activation]()]
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(latent_dim))

        layers += [nn.Linear(latent_dim, output_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
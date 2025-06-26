import torch
from torch import nn


class MlpEncoder(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1, num_layers: int = 2):
        super(MlpEncoder, self).__init__()

        layers = []

        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            # Input layer
            layers.append(nn.Linear(input_dim, hidden_dim))
            # Intermediate layers (if any)
            for i in range(num_layers - 2):
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
            # Output layer
            layers.append(nn.Linear(hidden_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor):
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            (batch_size, output_dim)
        """
        return self.mlp(x)

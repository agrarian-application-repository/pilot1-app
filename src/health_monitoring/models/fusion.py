import torch
from torch import nn


class ConcatLateFusionModule(nn.Module):

    def __init__(self, ts_dim: int, static_dim: int, output_dim: int):

        super(ConcatLateFusionModule, self).__init__()

        self.fusion = nn.Sequential(
            nn.Linear(ts_dim + static_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, ts_features: torch.FloatTensor, static_features: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            ts_features: (batch_size, ts_dim)
            static_features: (batch_size, static_dim)
        Returns:
            (batch_size, output_dim)
        """

        combined_features = torch.cat([ts_features, static_features], dim=1)
        return self.fusion(combined_features)


class GatedLateFusionModule(nn.Module):
    """Fuses time series and static features"""

    def __init__(self, ts_dim, static_dim, output_dim):
        super(GatedLateFusionModule, self).__init__()

        self.gate = nn.Sequential(
            nn.Linear(ts_dim + static_dim, output_dim),
            nn.Tanh(),
            nn.Linear(output_dim, 2),
            nn.Softmax(dim=1)
        )
        self.ts_proj = nn.Linear(ts_dim, output_dim)
        self.static_proj = nn.Linear(static_dim, output_dim)

    def forward(self, ts_features: torch.FloatTensor, static_features: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            ts_features: (batch_size, ts_dim)
            static_features: (batch_size, static_dim)
        Returns:
            (batch_size, output_dim)
        """

        combined = torch.cat([ts_features, static_features], dim=1)

        gates = self.gate(combined)  # (batch, 2)
        ts_weight = gates[:, 0:1]
        static_weight = gates[:, 1:2]

        ts_proj = self.ts_proj(ts_features)
        static_proj = self.static_proj(static_features)

        return ts_weight * ts_proj + static_weight * static_proj

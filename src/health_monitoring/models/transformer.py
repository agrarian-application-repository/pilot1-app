import torch
from torch import nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)


class TransformerTimeSeriesEncoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        num_layers: int = 2,
        num_heads: int = 8,
        pooling_type: str = 'attention'
    ):

        super(TransformerTimeSeriesEncoder, self).__init__()

        self.hidden_dim = hidden_dim

        self.pos_encoding = PositionalEncoding(input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.projection = nn.Linear(input_dim, hidden_dim)

        self.pooling_type = pooling_type  # 'last', 'mean', 'max', 'attention'
        if self.pooling_type == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            (batch_size, hidden_dim)
        """

        x = self.pos_encoding(x)
        output = self.encoder(x)  # (B, seq_len, input_dim)
        output = self.projection(output)  # (B, seq_len, hidden_dim)

        if self.pooling_type == 'mean':
            return output.mean(dim=1)
        elif self.pooling_type == 'attention':
            attention_weights = self.attention(output)
            attention_weights = F.softmax(attention_weights, dim=1)
            return (output * attention_weights).sum(dim=1)

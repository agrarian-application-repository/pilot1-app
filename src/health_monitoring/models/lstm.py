import torch
from torch import nn
import torch.nn.functional as F


class LSTMTimeSeriesEncoder(nn.Module):

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            dropout: float = 0.1,
            num_layers: int = 2,
            pooling_type: str = 'attention'
    ):
        super(LSTMTimeSeriesEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.pooling_type = pooling_type  # 'last', 'mean', 'max', 'attention'

        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

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
        output, (hidden, _) = self.encoder(x)  # output: (B, seq_len, hidden_dim)

        if self.pooling_type == 'last':
            return hidden[-1]  # Use last hidden state
        elif self.pooling_type == 'mean':
            return output.mean(dim=1)
        elif self.pooling_type == 'max':
            return output.max(dim=1)[0]
        elif self.pooling_type == 'attention':
            # Attention pooling
            attention_weights = self.attention(output)  # (B, seq_len, 1)
            attention_weights = F.softmax(attention_weights, dim=1)
            return (output * attention_weights).sum(dim=1)

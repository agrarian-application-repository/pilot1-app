from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool


class EntityGNN(nn.Module):
    """Graph Neural Network for entity relationships"""

    def __init__(self, input_dim, hidden_dim, output_dim, gnn_type='gcn', num_layers=2):
        super().__init__()
        self.gnn_type = gnn_type
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer
        if gnn_type == 'gcn':
            self.convs.append(GCNConv(input_dim, hidden_dim))
        elif gnn_type == 'gat':
            self.convs.append(GATConv(input_dim, hidden_dim, heads=4, concat=False))

        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            if gnn_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == 'gat':
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Output layer
        if num_layers > 1:
            if gnn_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim, output_dim))
            elif gnn_type == 'gat':
                self.convs.append(GATConv(hidden_dim, output_dim, heads=1, concat=False))

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index, batch=None):
        """
        Args:
            x: (num_nodes, input_dim)
            edge_index: (2, num_edges)
            batch: batch indicator for graph-level tasks
        Returns:
            (num_nodes, output_dim) or (num_graphs, output_dim)
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            if i < len(self.convs) - 1:  # Not the last layer
                if i < len(self.batch_norms):
                    x = self.batch_norms[i](x)
                x = F.relu(x)
                x = self.dropout(x)

        # If batch is provided, perform graph-level pooling
        if batch is not None:
            x = global_mean_pool(x, batch)

        return x
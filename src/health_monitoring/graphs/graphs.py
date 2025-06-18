import numpy as np
from torch_geometric.nn import radius_graph, knn_graph
import torch


def create_knn_graph(tracked_positions, k=5):
    """
    Create a k-Nearest Neighbors (k-NN) graph.

    Parameters:
        tracked_positions (Tensor): (N, 2) tensor containing node positions.
        k (int): Number of nearest neighbors to connect each node to.

    Returns:
        edge_index (Tensor): (2, E) tensor defining the edges of the graph.
    """
    edge_index = knn_graph(tracked_positions, k=k, loop=False)
    return edge_index


def create_radius_graph(tracked_positions, r=0.1):
    """
    Create a radius-based graph where edges are created between nodes within a given radius.

    Parameters:
        tracked_positions (Tensor): (N, 2) tensor containing node positions.
        r (float): Radius within which nodes are connected.

    Returns:
        edge_index (Tensor): (2, E) tensor defining the edges of the graph.
    """
    edge_index = radius_graph(tracked_positions, r=r, loop=False)
    return edge_index


def create_edges(num_nodes, distance_matrix):
    # Create edge index and edge attributes
    edge_index = []
    edge_attr = []

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # Avoid duplicate edges
            edge_index.append([i, j])
            edge_index.append([j, i])  # Ensure undirected edges
            edge_attr.append(distance_matrix[i, j])
            edge_attr.append(distance_matrix[i, j])  # Duplicate for symmetry

    edge_index = torch.tensor(edge_index, dtype=torch.long).T  # Shape (2, num_edges)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1)  # Shape (num_edges, 1)

    return edge_index, edge_attr


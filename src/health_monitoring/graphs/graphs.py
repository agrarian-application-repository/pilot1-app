import numpy as np
from torch_geometric.nn import radius_graph, knn_graph
from scipy.spatial.distance import cdist
import torch


def create_distance_matrix(currently_tracked_data):

    nodes_ids = list(currently_tracked_data.keys())  # List of entity IDs

    # Extract the last valid coordinates for each node
    last_coords = []
    for id in nodes_ids:
        # all ids must be valid at last point in time t, because are the one see by the tracker at that frame
        if currently_tracked_data[id]["valid"][0, -1] is False:
            raise ValueError(f"Last value should be valid, but is not. ID: {id}, validity: {currently_tracked_data[id]['valid']}")

        last_coords.append(currently_tracked_data[id]["coords"][:, -1])  # Extract last x_norm/y_norm
        # y_norm = y_norm_yolo * aspect_ratio to preserve spatial relationship in non-square frame yolo detection
        # for 16:9, x varies in [0,1], y varies in [0,9/16]
        # such that a distance of 1 pixel on the x is equal to a distance of 1 pixel on the y

    last_coords = np.array(last_coords)  # Shape: (num_nodes, 2)

    # Compute pairwise distances
    distance_matrix = cdist(last_coords, last_coords, metric='euclidean')
    return distance_matrix


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


def create_fully_connected_graph(num_nodes):
    """
    Create a fully connected graph where each node is connected to every other node.

    Parameters:
        num_nodes (int): Number of nodes in the graph.

    Returns:
        edge_index (Tensor): (2, E) tensor defining the edges of the graph.
    """
    row = torch.arange(num_nodes).repeat(num_nodes)
    col = torch.arange(num_nodes).repeat_interleave(num_nodes)
    edge_index = torch.stack([row, col], dim=0)

    # Remove self-loops
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]

    return edge_index

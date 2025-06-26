import torch
from torch_geometric.data import Data

from src.health_monitoring.graphs.graphs import create_knn_graph, create_radius_graph


def create_pyg_graph(
        graph_mode: str,
        graph_param: int|float,
        pos,
):

    if graph_mode == "knn":
        edge_index = create_knn_graph(pos, k=graph_param)

    elif graph_mode == "radius":
        edge_index = create_radius_graph(pos, r=graph_param)

    else:
        raise ValueError("'graph_mode' should only be 'knn' or 'radius'")

    return edge_index


def create_pyg_dataset(
        graph_mode,
        graph_param,
        ids,
        features,
):

    num_nodes = len(ids)

    edge_index = create_pyg_graph(
        graph_mode,
        graph_param,
        features["pos"],
    )

    edge_attr = create_edge_attributes()

    # Create PyG Data object
    data = Data(
        ids=torch.tensor(ids, dtype=torch.int),
        pos=torch.tensor(features["pos"], dtype=torch.float),
        valid=torch.tensor(features["valid"], dtype=torch.float),
        ts=torch.tensor(features["ts"], dtype=torch.float),
        non_ts=torch.tensor(features["non_ts"], dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )

    return data







import torch
from torch_geometric.data import Data

from src.drone_utils.flight_logs import parse_drone_flight_data
from src.drone_utils.gsd import get_meters_per_pixel

from src.health_monitoring.graphs.graphs import create_knn_graph, create_radius_graph, create_fully_connected_graph


def create_pyg_graph(
        num_nodes,
        tracked_positions,
        input_args,
        drone_args,
        flight_data_file,
        frame_id,
        frame_width,
        frame_height,
):

    if input_args["graph_strategy"] == "knn":
        edge_index = create_knn_graph(tracked_positions, k=input_args["graph_knn_neighbours"])

    elif input_args["graph_strategy"] == "radius":

        # load frame flight data to extract drone elevation
        flight_frame_data = parse_drone_flight_data(flight_data_file, frame_id)

        # compute GSR (ground sampling resolution) = meters/pixel given drone elevation and camera params
        meters_per_pixel = get_meters_per_pixel(
            rel_altitude_m=flight_frame_data["rel_alt"],
            focal_length_mm=drone_args["true_focal_len_mm"],
            sensor_width_mm=drone_args["sensor_width_mm"],
            sensor_height_mm=drone_args["sensor_height_mm"],
            sensor_width_pixels=drone_args["sensor_width_pixels"],
            sensor_height_pixels=drone_args["sensor_height_pixels"],
            image_width_pixels=frame_width,
            image_height_pixels=frame_height,
        )

        # compute the size in meters of the frame's width
        frame_width_m = frame_width * meters_per_pixel

        # radius is a value in [0.0, 1.0]
        # is computed as the proportion between the size in meters specified by the user
        # over the width of the frame
        # this accounts for drone changes in elevation:
        # - if the drone climbs the radius will get smaller over the frame (size in meters doesn't change)
        # - if the drone descends the radius will get larger over the frame (size in meters doesn't change)
        radius = max(1.0, (input_args["graph_radius_meters"] / frame_width_m))

        edge_index = create_radius_graph(tracked_positions, r=radius)

    else:
        edge_index = create_fully_connected_graph(num_nodes)

    return edge_index


def create_pyg_dataset(
        history,
        input_args,
        drone_args,
        flight_data_file,
        frame_id,
        frame_width,
        frame_height,
):

    num_nodes = len(history.last_ids_list)

    tracked_ids, tracked_positions = history.get_last_update_arrays()
    # (N,) array and (N,2)=(x,y) array

    edge_index = create_pyg_graph(
        num_nodes,
        tracked_positions,
        input_args,
        drone_args,
        flight_data_file,
        frame_id,
        frame_width,
        frame_height,
    )

    pos, valid = history.get_and_aggregate_ids_history(history.last_ids_list)

    # Create PyG Data object
    data = Data(
        ids=torch.tensor(tracked_ids, dtype=torch.int),
        pos=torch.tensor(pos, dtype=torch.float),  # sequence of object positions (Nnodes, x_norm & y_norm, TSlenght) = (N, 2,T)
        valid=torch.tensor(valid, dtype=torch.float),   # sequence of validities (Nnodes, 1, TSlenght) = (N, 1, T)
        edge_index=edge_index,
        edge_attr=edge_attr,
    )

    return data







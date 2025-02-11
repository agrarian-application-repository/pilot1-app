import cv2
import torch
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph, knn_graph
from scipy.spatial.distance import cdist
import numpy as np
from pathlib import Path

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
PURPLE = (128, 0, 128)
CLASS_COLOR = [BLUE, PURPLE]


class HistoryTracker:
    """
     the object should act as follows. when  initialized, the input argument should be a lenght value.
     each key should comprise of two arrays, a list of value couples (x,y), an a boolean array indicating wheter the tuple at that position is valid.
     The calss should also have an update function that, given a list of ids and the corresponding coordinates, inserts the tuple in the array and sets it as valid.
     all other keys that are stored by the object but do not appear in the provided new set of id/tuples pairs, should replicate the last value and set the boolean mask value as False to indicate the data is artificial.
     If a key that never appeared before appears in the update, that key is added and a list/mask of lenght equal to the initially specified lenght argument is created for that key, with all values in the mask excet the last one being false, and the list of couples is comprised of the same tuple repeated LEN times.
     Finally, at each update step the oldest value in the list of tuples/masks should be dropped similarly to a fifo queue where LEN acts as the queu lenght)
    """

    def __init__(self, window_size):
        # window_size indicates the size of the value and mask arrays
        self.window_size = window_size
        self.data = {}

        self.last_ids_list = []
        self.last_positions_list = []

    def update(self, ids, coordinates):

        """
        Update the dictionary with new tuples and boolean masks.
        - ids: List of keys to update
        - coordinates: List of (x, y) tuples corresponding to the ids
        """

        # Iterate over the ids and coordinates provided in the update
        for i, (key, coords) in enumerate(zip(ids, coordinates)):

            if key not in self.data:
                # If the key is new, initialize its tuple list and mask array
                self.data[key] = {}
                self.data[key]["coords"] = [coords] * self.window_size
                self.data[key]["valid"] = [False] * (self.window_size - 1) + [True]
            else:
                # If the key already exists:
                # Add the new value and set the validity indicator to True
                self.data[key]["coords"].append(coords)
                self.data[key]["valid"].append(True)
                # And remove the oldest value to maintain the window size
                del self.data[key]["coords"][0]
                del self.data[key]["valid"][0]

        non_updated_keys = set(self.data.keys()) - set(ids)

        for key in non_updated_keys:
            # Repeat the last value, setting validity indicator to False
            last_value = self.data[key]["coords"][-1]
            self.data[key]["coords"].append(last_value)
            self.data[key]["valid"].append(False)
            # And remove the oldest value to maintain the window size
            del self.data[key]["coords"][0]
            del self.data[key]["valid"][0]

        # save the set of ids and positions contributing to the last update
        self.last_ids_list = ids
        self.last_positions_list = coordinates

    def get_ids_history(self, ids):
        # ids must already exist
        if not set(ids).issubset(set(self.data.keys())):
            raise ValueError(f"Cannot get history of an id that does not exists. requested: {set(ids)}, existing: {set(self.data.keys())}")

        tracked = {}
        for id in ids:
            tracked[id] = {}
            tracked[id]["coords"] = np.array(self.data[id]["coords"]).T     # (2, TSlenght)
            tracked[id]["valid"] = np.array(self.data[id]["valid"]).reshape(1, -1)  # (1, TSlenght)

        return tracked

    def get_and_aggregate_ids_history(self, ids):
        """
        Retrieves and aggregates time series data for given IDs.

        Parameters:
            ids (list): List of IDs to fetch history for.

        Returns:
            coords_array (numpy.ndarray): (N, 2, TS_length) array of coordinates.
            valid_array (numpy.ndarray): (N, 1, TS_length) array of validity flags.
        """
        tracked = self.get_ids_history(ids)  # Get individual histories

        # Extract coordinate and validity arrays
        coords_list = [tracked[id]["coords"] for id in ids]  # List of (2, TS_length) arrays
        valid_list = [tracked[id]["valid"] for id in ids]  # List of (1, TS_length) arrays

        # Convert lists to numpy arrays of shape (N, 2, TS_length) and (N, 1, TS_length)
        aggregated_coords_array = np.stack(coords_list, axis=0)  # Shape: (N, 2, TS_length)
        aggregated_valid_array = np.stack(valid_list, axis=0)  # Shape: (N, 1, TS_length)

        return aggregated_coords_array, aggregated_valid_array

    def get_last_update_arrays(self):
        last_ids_array = np.array(self.last_ids_list)   # (N, )
        last_positions_array = np.array(self.last_positions_list)   # (N,2)
        return last_ids_array, last_positions_array


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
        # for 16:9, x varies in [0,1], y varies in [0,16/9] such that a distance of 1 pixel on the x is equal to a distance of 1 pixel on the y

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


# TODO check if breaks because of differemnt aspect ratio
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


# TODO check if breaks because of differemnt aspect ratio
def meters_to_normalized_radius(radius_m, frame_width_m):
    """
    Converts a radius in meters to the normalized YOLO space.

    Parameters:
        radius_m (float): Radius in meters.
        frame_width_m (float): Frame width in meters.

    Returns:
        float: Normalized radius for YOLO.
    """

    normalized_radius = max(radius_m / frame_width_m, 1.0)

    return normalized_radius


def create_pyg_dataset(history, strategy, strategy_value=None):

    num_nodes = len(history.last_ids_list)

    tracked_ids, tracked_positions = history.get_last_update_arrays()
    # (N,) array and (N,2)=(x,y) array

    if strategy == "radius":
        edge_index = create_knn_graph(tracked_positions, k=strategy_value)
    elif strategy == "knn":
        edge_index = create_radius_graph(tracked_positions, r=strategy_value) # TODO x,y different_scales
    else:
        edge_index = create_fully_connected_graph(num_nodes)

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


def perform_anomaly_detection(anomaly_detector, history: HistoryTracker, anomaly_detection_args):

    dataset = create_pyg_dataset(history, strategy, strategy_value)

    status = anomaly_detector(dataset)

    return status


def perform_tracking(detector, history: HistoryTracker, frame, tracking_args, aspect_ratio):
    # track animals in frame
    tracking_results = detector.track(source=frame, stream=True, persist=True, **tracking_args)

    # Parse detection results to get bounding boxes & create additional variables to store useful inf
    classes = tracking_results[0].boxes.cls.cpu().numpy().astype(int)
    xywh_boxes = tracking_results[0].boxes.xywh.cpu().numpy().astype(int)
    xyxy_boxes = tracking_results[0].boxes.xyxy.cpu().numpy().astype(int)

    boxes_centers = xywh_boxes[:, :2]
    boxes_corner1 = xyxy_boxes[:, :2]
    boxes_corner2 = xyxy_boxes[:, 2:]

    xywhn_boxes = tracking_results[0].boxes.xywhn.cpu().numpy()
    normalized_boxes_centers = xywhn_boxes[:, :2]
    normalized_boxes_centers[:1] = normalized_boxes_centers[:1] * aspect_ratio
    normalized_boxes_centers = normalized_boxes_centers.astype(int)

    # Parse tracking ID
    if tracking_results[0].boxes.id is not None:
        ids_list = tracking_results[0].boxes.id.int().cpu().tolist()
        positions_list = normalized_boxes_centers.tolist()
    else:   # TODO is no tracking = no detections? (no)
        ids_list = []
        positions_list = []

    history.update(ids_list, positions_list)

    return classes, boxes_centers, normalized_boxes_centers, boxes_corner1, boxes_corner2


def send_alert(alerts_file, frame_id: int, num_anomalies: int):
    # Write alert to file
    alerts_file.write(f"Alert: Frame {frame_id} - {num_anomalies} instances of anomalous behaviour detected.\n")


def draw_detections(
        annotated_frame,
        classes,
        are_anomalous,
        boxes_corner1,
        boxes_corner2,
):
    # drawing safety circles & detection boxes
    for obj_class, is_anomaly, box_corner1, box_corner2 in zip(classes, are_anomalous, boxes_corner1, boxes_corner2):
        # Choose color depending on class (blue sheep, purple goat) and it being an anomaly (red)
        color = RED if is_anomaly else CLASS_COLOR[obj_class]
        # Draw bounding box on frame
        cv2.rectangle(annotated_frame, box_corner1, box_corner2, color, 2)


def annotate_video(
        output_dir,
        annotated_writer,
        frame,
        frame_id,
        cooldown_has_passed,
        anomaly_exists,
        classes,
        are_anomalous,
        boxes_corner1,
        boxes_corner2,
):
    annotated_frame = frame.copy()  # copy of the original frame on which to draw
    draw_detections(annotated_frame, classes, are_anomalous, boxes_corner1, boxes_corner2)

    # save the annotated frame to the video
    annotated_writer.write(annotated_frame)
    if cooldown_has_passed and anomaly_exists:
        # save also independent frame for improved insight
        annotated_img_path = Path(output_dir, f"anomaly_frame_{frame_id}_annotated.png")
        cv2.imwrite(annotated_img_path, annotated_frame)

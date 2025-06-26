from src.health_monitoring.tracking.history import HistoryTracker
from src.health_monitoring.graphs.dataset import create_pyg_dataset
from src.health_monitoring.anomaly_detection.statistical_methods import detect_anomalies_statistical
from src.health_monitoring.anomaly_detection.history_to_features import collect_features

from src.drone_utils.gsd import get_meters_per_pixel
from src.drone_utils.flight_logs import parse_drone_flight_data


def compute_gsd_from_file(
    drone_args,
    flight_data_file,
    frame_id,
    frame_width,
    frame_height,
):
    # load frame flight data
    frame_flight_data = parse_drone_flight_data(flight_data_file, frame_id)
    # Perform the pixels to meters conversion using the sensor resolution
    meters_per_pixel = get_meters_per_pixel(
        rel_altitude_m=frame_flight_data["rel_alt"],
        focal_length_mm=drone_args["true_focal_len_mm"],
        sensor_width_mm=drone_args["sensor_width_mm"],
        sensor_height_mm=drone_args["sensor_height_mm"],
        sensor_width_pixels=drone_args["sensor_width_pixels"],
        sensor_height_pixels=drone_args["sensor_height_pixels"],
        image_width_pixels=frame_width,
        image_height_pixels=frame_height,
    )
    return meters_per_pixel


def compute_area_fraction(
        radius_meters: float,
        meters_per_pixel: float,
        frame_width: int,
) -> float:
    # compute a fraction of ara where to search for neighbours based on a distance in meters
    area_fraction = (radius_meters / meters_per_pixel) / frame_width
    return area_fraction


# when not computing the anomaly detection
# transpose last computed anomaly status onto current detections
def merge_previous_anomaly_status_current_detections(
    current_ids: list[int],
    previous_ids: list[int],
    previous_anomaly_status: list[bool],
) -> list[bool]:

    assert len(previous_ids) == len(previous_anomaly_status)
    # Build a dictionary table mapping previous ids to corresponding anomaly status
    previous_id_mask_mapping = dict(zip(previous_ids, previous_anomaly_status))
    # Build a list of current anomaly status from the current list of ids and the old anomaly statuses
    # entities previously not present are set to False (no anomaly) by default
    current_anomaly_status = [previous_id_mask_mapping.get(id_, False) for id_ in current_ids]

    return current_anomaly_status   # (N, ) of bool values


def perform_anomaly_detection_statistical(
    history: HistoryTracker,
    radius_r,
    knn_k,
    majority_vote_threshold,
    current_ids_list,
    frame_id,
) -> list[bool]:

    # generate statistical features from processing of animals timeseries
    features = collect_features(history, current_ids_list, radius_r, knn_k, use_stats=True)
    # create and apply an ensemble of statistical and ML models to detect anomalies
    anomaly_status, _ = detect_anomalies_statistical(features, frame_id, majority_vote_threshold)
    # (N, ) of bool values

    return anomaly_status


"""
def perform_anomaly_detection_graph(
        anomaly_detector,
        anomaly_detection_args,
        history: HistoryTracker,
        graph_mode: str,
        knn_k,
        radius_meters,
        current_ids_list,
        drone_args,
        flight_data_file,
        frame_id,
        frame_width,
        frame_height,
):

    area_knn_meters,
    knn_k,
    current_ids_list,
    drone_args,
    flight_data_file,
    frame_id,
    frame_width,
    frame_height,

    # transform radius from meters to fraction of the image width based on drone flight altitude
    radius_frac = compute_area_fraction(
        radius_meters,
        drone_args,
        flight_data_file,
        frame_id,
        frame_width,
        frame_height,
    )

    graph_param = knn_k if graph_mode == "knn" else radius_frac

    # generate dictionary of features from processing of animals timeseries
    features = collect_features(history, current_ids_list, radius_frac, knn_k, use_stats=False)

    dataset = create_pyg_dataset(graph_mode, graph_param, ids, features)

    anomaly_status = anomaly_detector(dataset)
    return anomaly_status   # (N, ) of bool values
"""


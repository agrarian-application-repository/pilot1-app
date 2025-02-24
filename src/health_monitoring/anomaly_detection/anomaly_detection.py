from src.health_monitoring.tracking.history import HistoryTracker
from src.health_monitoring.graphs.dataset import create_pyg_dataset


def perform_anomaly_detection(
        anomaly_detector,
        anomaly_detection_args,
        history: HistoryTracker,
        input_args,
        drone_args,
        flight_data_file,
        frame_id,
        frame_width,
        frame_height,
):
    dataset = create_pyg_dataset(history, input_args, drone_args, flight_data_file, frame_id, frame_width, frame_height)
    status = anomaly_detector(dataset)
    return status

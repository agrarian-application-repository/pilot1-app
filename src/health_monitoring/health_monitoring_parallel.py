from typing import Any

import multiprocessing as mp

from src.health_monitoring.processes.reading import VideoReader
from src.health_monitoring.processes.tracking import TrackingWorker
from src.health_monitoring.processes.anomaly import AnomalyDetectionWorker
from src.health_monitoring.processes.annotation import AnnotationWorker
from src.health_monitoring.processes.output import VideoStreamWriter, NotificationsStreamWriter


def perform_health_monitoring_analysis(
        input_args: dict[str:Any],
        output_args: dict[str:Any],
        tracking_args: dict[str:Any],
        anomaly_detection_args: dict[str:Any],
        drone_args: dict[str:Any],
) -> None:

    # ============== SETUP QUEUES AND EVENTS ===================================

    shared_dict = mp.Manager().dict()
    video_info_set_event = mp.Event()

    tracking_in_queue = mp.Queue()

    tracking_results_queue = mp.Queue()

    anomaly_detection_results_queue = mp.Queue()

    video_stream_queue = mp.Queue()
    notifications_stream_queue = mp.Queue()
    stream_queues = [video_stream_queue, notifications_stream_queue]

    # ============== START PROCESSES PROCESSES ===================================

    # Create VideoReader process
    video_reader_process = VideoReader(
        source=input_args["source"],
        result_queue=tracking_in_queue,
        shared_dict=shared_dict,
        video_info_set_event=video_info_set_event,
    )
    video_reader_process.start()
    video_info_set_event.wait()

    # Create TrackingWorker process
    tracking_process = TrackingWorker(
        detection_args=detection_args,
        input_queue=tracking_in_queue,
        result_queue=tracking_results_queue
    )
    tracking_process.start()

    # Create AnomalyIdentification process
    anomaly_identification_process = AnomalyDetectionWorker(
        input_queue=tracking_results_queue,
        result_queue=anomaly_detection_results_queue,
        shared_dict=shared_dict,
    )
    anomaly_identification_process.start()

    # Create VideoAnnotatorWorker
    annotation_process = AnnotationWorker(
        input_queue=anomaly_detection_results_queue,
        stream_queues=stream_queues,
        shared_dict=shared_dict,
    )
    annotation_process.start()

    # Create VideoStreamWriter process
    video_writer_process = VideoStreamWriter(
        output_dir=output_args["output_dir"],
        input_queue=video_stream_queue,
        shared_dict=shared_dict,
    )
    video_writer_process.start()

    # Create VideoStreamWriter process
    notification_writer_process = NotificationsStreamWriter(
        output_dir=output_args["output_dir"],
        cooldown_seconds=input_args["alerts_cooldown_seconds"],
        input_queue=notifications_stream_queue,
        shared_dict=shared_dict,
    )
    notification_writer_process.start()

    # ============== WAIT FOR PROCESSES TO FINISH AND RELEASE RESOURCES ===================================

    video_reader_process.join()
    tracking_process.join()
    anomaly_identification_process.join()
    annotation_process.join()
    video_writer_process.join()
    notification_writer_process.join()

    tracking_in_queue.close()
    tracking_results_queue.close()
    anomaly_detection_results_queue.close()
    video_stream_queue.close()
    notifications_stream_queue.close()

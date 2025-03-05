from typing import Any

import multiprocessing as mp

from src.in_danger.processes.reading import VideoReader
from src.in_danger.processes.detection import DetectionWorker
from src.in_danger.processes.segmentation import SegmentationWorker
from src.in_danger.processes.geo import GeoWorker
from src.in_danger.processes.danger import DangerDetectionWorker
from src.in_danger.processes.annotation import AnnotationWorker
from src.in_danger.processes.output import VideoStreamWriter, NotificationsStreamWriter


def perform_in_danger_analysis(
        input_args: dict[str:Any],
        output_args: dict[str:Any],
        detection_args: dict[str:Any],
        segmentation_args: dict[str:Any],
        drone_args: dict[str:Any],
) -> None:

    # ============== SETUP QUEUES AND EVENTS ===================================

    shared_dict = mp.Manager().dict()
    video_info_set_event = mp.Event()

    detection_in_queue = mp.Queue()
    segmentation_in_queue = mp.Queue()
    geo_in_queue = mp.Queue()
    models_in_queues = [detection_in_queue, segmentation_in_queue, geo_in_queue]

    detection_results_queue = mp.Queue()
    segmentation_results_queue = mp.Queue()
    geo_results_queue = mp.Queue()
    models_results_queues = [detection_results_queue, segmentation_results_queue, geo_results_queue]

    danger_detection_results_queue = mp.Queue()

    video_stream_queue = mp.Queue()
    notifications_stream_queue = mp.Queue()
    stream_queues = [video_stream_queue, notifications_stream_queue]

    # ============== START PROCESSES PROCESSES ===================================

    # Create VideoReader process
    video_reader_process = VideoReader(
        source=input_args["source"],
        models_queues=models_in_queues,
        shared_dict=shared_dict,
        video_info_set_event=video_info_set_event,
    )
    video_reader_process.start()
    video_info_set_event.wait()

    # Create DetectionWorker process
    detection_process = DetectionWorker(
        detection_args=detection_args,
        input_queue=detection_in_queue,
        result_queue=detection_results_queue
    )
    detection_process.start()

    # Create DetectionWorker process
    segmentation_process = SegmentationWorker(
        segmentation_args=segmentation_args,
        input_queue=segmentation_in_queue,
        result_queue=segmentation_results_queue
    )
    segmentation_process.start()

    # Create GeoWorker process
    geo_process = GeoWorker(
        input_args=input_args,
        drone_args=drone_args,
        input_queue=geo_in_queue,
        result_queue=geo_results_queue,
        shared_dict=shared_dict,
    )
    geo_process.start()

    # Create DangerIdentification process
    danger_identification_process = DangerDetectionWorker(
        models_results_queues=models_results_queues,
        result_queue=danger_detection_results_queue,
        shared_dict=shared_dict,
    )
    danger_identification_process.start()

    # Create VideoAnnotatorWorker
    annotation_process = AnnotationWorker(
        input_queue=danger_detection_results_queue,
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
    detection_process.join()
    segmentation_process.join()
    geo_process.join()
    danger_identification_process.join()
    annotation_process.join()
    video_writer_process.join()
    notification_writer_process.join()

    detection_in_queue.close()
    segmentation_in_queue.close()
    geo_in_queue.close()
    detection_results_queue.close()
    segmentation_results_queue.close()
    geo_results_queue.close()
    danger_detection_results_queue.close()
    video_stream_queue.close()
    notifications_stream_queue.close()

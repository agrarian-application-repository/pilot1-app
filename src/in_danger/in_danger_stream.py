from typing import Any

import multiprocessing as mp

from src.in_danger.processes.detection import DetectionWorker
from src.in_danger.processes.segmentation import SegmentationWorker
from src.in_danger.processes.geo import GeoWorker
from src.in_danger.processes.danger import DangerDetectionWorker
from src.in_danger.processes.annotation import AnnotationWorker

from src.shared.processes.stream_video_reader import StreamVideoReader
from src.shared.processes.stream_telemetry_listener import StreamTelemetryListener
from src.shared.processes.frame_telemetry_combiner import FrameTelemetryCombiner
from src.shared.processes.output_video_streamer import VideoStreamWriter
from src.shared.processes.output_alert_streamer import NotificationsStreamWriter


def perform_in_danger_analysis(
        input_args: dict[str, Any],
        output_args: dict[str, Any],
        detection_args: dict[str, Any],
        segmentation_args: dict[str, Any],
        drone_args: dict[str, Any],
) -> None:
    
    # TODO: integrate additional args
    video_info_dict = {
        "frame_width": 1920,
        "frame_height": 1080,
        "fps": 30,
    }
    
    telemetry_in_port=12345
    video_url_out = "rtsp://127.0.0.1:8554/annot"
    alerts_url_out = "tcp://127.0.0.1:54321"

    input_args["video_info_dict"] = video_info_dict
    output_args["telemetry_in_port"] = telemetry_in_port
    output_args["video_url_out"] = video_url_out
    output_args["alerts_url_out"] = alerts_url_out

    # ============== SETUP QUEUES AND EVENTS ===================================
    


    received_frames_queue = mp.Queue()
    received_telemetries_queue = mp.Queue()

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

    # Create StreamVideoReader process
    video_reader_process = StreamVideoReader(
        video_info_dict=input_args["video_info_dict"],
        source=input_args["source"],
        frame_queue=received_frames_queue,
    )

    # Create StreamTelemetryReader process
    telemetry_reader_process = StreamTelemetryListener(
        port=output_args["telemetry_in_port"],
        telemetry_queue=received_telemetries_queue,
    )

    # Create Frame-Telemetry combiner process
    combiner_process = FrameTelemetryCombiner(
        frame_queue=received_frames_queue,
        telemetry_queue=received_telemetries_queue,
        output_queues=models_in_queues,
    )

    # Create DetectionWorker process
    detection_process = DetectionWorker(
        detection_args=detection_args,
        input_queue=detection_in_queue,
        result_queue=detection_results_queue
    )

    # Create DetectionWorker process
    segmentation_process = SegmentationWorker(
        segmentation_args=segmentation_args,
        input_queue=segmentation_in_queue,
        result_queue=segmentation_results_queue
    )

    # Create GeoWorker process
    geo_process = GeoWorker(
        input_args=input_args,
        drone_args=drone_args,
        input_queue=geo_in_queue,
        result_queue=geo_results_queue,
        video_info_dict=input_args["video_info_dict"],
    )

    # Create DangerIdentification process
    danger_identification_process = DangerDetectionWorker(
        models_results_queues=models_results_queues,
        result_queue=danger_detection_results_queue,
        video_info_dict=input_args["video_info_dict"],
    )

    # Create VideoAnnotatorWorker
    annotation_process = AnnotationWorker(
        input_queue=danger_detection_results_queue,
        stream_queues=stream_queues,
        video_info_dict=input_args["video_info_dict"],
    )

    # Create VideoStreamWriter process
    video_writer_process = VideoStreamWriter(
        video_info_dict=input_args["video_info_dict"],
        input_queue=video_stream_queue,
        output_dir=output_args["output_dir"],
        output_url=output_args["video_url_out"],
    )
    video_writer_process.start()

    # Create VideoStreamWriter process
    notification_writer_process = NotificationsStreamWriter(
        video_info_dict=input_args["video_info_dict"],
        cooldown_seconds=input_args["alerts_cooldown_seconds"],
        input_queue=notifications_stream_queue,
        output_dir=output_args["output_dir"],
        output_url=output_args["alerts_url_out"]
    )
    notification_writer_process.start()

    # ============== START PROCESSES ===================================
    
    video_reader_process.start()
    telemetry_reader_process.start()
    combiner_process.start()
    detection_process.start()
    segmentation_process.start()
    geo_process.start()
    danger_identification_process.start()
    annotation_process.start()


    # ============== WAIT FOR PROCESSES TO FINISH AND RELEASE RESOURCES ===================================

    video_reader_process.join()
    telemetry_reader_process.stop() # stop telemetry listener when video reading stops
    telemetry_reader_process.join()
    
    detection_process.join()
    segmentation_process.join()
    geo_process.join()
    danger_identification_process.join()
    annotation_process.join()
    
    video_writer_process.join()
    notification_writer_process.join()


    received_frames_queue.close()
    received_telemetries_queue.close()
    detection_in_queue.close()
    segmentation_in_queue.close()
    geo_in_queue.close()
    detection_results_queue.close()
    segmentation_results_queue.close()
    geo_results_queue.close()
    danger_detection_results_queue.close()
    video_stream_queue.close()
    notifications_stream_queue.close()

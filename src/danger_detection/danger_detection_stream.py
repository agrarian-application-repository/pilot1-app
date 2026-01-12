from typing import Any
import os

import multiprocessing as mp

from src.danger_detection.processes.detection import DetectionWorker
from src.danger_detection.processes.segmentation import SegmentationWorker
from src.danger_detection.processes.geo import GeoWorker
from src.danger_detection.processes.danger import DangerDetectionWorker
from src.danger_detection.processes.annotation import AnnotationWorker

from src.shared.processes.stream_video_reader import StreamVideoReader
from src.shared.processes.mqtt_telemetry_listener import MqttCollectorProcess
from src.shared.processes.frame_telemetry_combiner import FrameTelemetryCombiner
from src.shared.processes.output_video_streamer import VideoProducerProcess
from src.shared.processes.output_alert_streamer import NotificationsStreamWriter
from src.shared.processes.video_storage_manager import VideoPersistenceProcess

from src.shared.processes.constants import *


def fetch_env(key, default):
    """Fetches env var; returns default if env var is missing or empty."""
    # TODO handle case default is None or it's fine that the val be set to none
    val = os.environ.get(key)
    # if val is not set, return the default value
    if val is None or val.strip() == "":
        return default
    # if val is set, auto-cast to default type
    return type(default)(val)


def perform_danger_detection(
        input_args: dict[str, Any],
        output_args: dict[str, Any],
        detection_args: dict[str, Any],
        segmentation_args: dict[str, Any],
        drone_args: dict[str, Any],
) -> None:

    # ============== SETUP EVENTS ===================================

    stop_event = mp.Event()
    error_event = mp.Event()

    # ============== SETUP QUEUES ===================================

    # Layer 1 -> 2
    max_size_frame_reader_out = fetch_env("MAX_SIZE_FRAME_READER_OUT", MAX_SIZE_FRAME_READER_OUT)
    max_size_telemetry_reader_out = fetch_env("MAX_SIZE_TELEMETRY_READER_OUT", MAX_SIZE_TELEMETRY_READER_OUT)
    # Layer 2 -> 3
    max_size_detection_in = fetch_env("MAX_SIZE_DETECTION_IN", MAX_SIZE_DETECTION_IN)
    max_size_segmentation_in = fetch_env("MAX_SIZE_SEGMENTATION_IN", MAX_SIZE_SEGMENTATION_IN)
    max_size_geo_in = fetch_env("MAX_SIZE_GEO_IN", MAX_SIZE_GEO_IN)
    # Layer 3 -> 4
    max_size_detection_results = fetch_env("MAX_SIZE_DETECTION_RESULTS", MAX_SIZE_DETECTION_RESULTS)
    max_size_segmentation_results = fetch_env("MAX_SIZE_SEGMENTATION_RESULTS", MAX_SIZE_SEGMENTATION_RESULTS)
    max_size_geo_results = fetch_env("MAX_SIZE_GEO_RESULTS", MAX_SIZE_GEO_RESULTS)
    # Layer 4 -> 5
    max_size_danger_detection_result = fetch_env("MAX_SIZE_DANGER_DETECTION_RESULT", MAX_SIZE_DANGER_DETECTION_RESULT)
    # Layer 5 -> 6
    max_size_video_stream = fetch_env("MAX_SIZE_VIDEO_STREAM", MAX_SIZE_VIDEO_STREAM)
    max_size_notifications_stream = fetch_env("MAX_SIZE_NOTIFICATIONS_STREAM", MAX_SIZE_NOTIFICATIONS_STREAM)
    # Layer 6 -> 7
    max_size_video_storage = fetch_env("MAX_SIZE_VIDEO_STORAGE", MAX_SIZE_VIDEO_STORAGE)


    # LAYER 1 -> 2
    frame_reader_out_queue = mp.Queue(maxsize=max_size_frame_reader_out)
    telemetry_reader_out_queue = mp.Queue(maxsize=max_size_telemetry_reader_out)
    # LAYER 2 -> 3
    detection_in_queue = mp.Queue(maxsize=max_size_detection_in)
    segmentation_in_queue = mp.Queue(maxsize=max_size_segmentation_in)
    geo_in_queue = mp.Queue(maxsize=max_size_geo_in)
    models_in_queues = [detection_in_queue, segmentation_in_queue, geo_in_queue]
    # LAYER 3 -> 4
    detection_results_queue = mp.Queue(maxsize=max_size_detection_results)
    segmentation_results_queue = mp.Queue(maxsize=max_size_segmentation_results)
    geo_results_queue = mp.Queue(maxsize=max_size_geo_results)
    models_results_queues = [detection_results_queue, segmentation_results_queue, geo_results_queue]
    # LAYER 4 -> 5
    danger_detection_result_queue = mp.Queue(maxsize=max_size_danger_detection_result)
    # LAYER 5 -> 6
    video_stream_queue = mp.Queue(maxsize=max_size_video_stream)
    notifications_stream_queue = mp.Queue(maxsize=max_size_notifications_stream)
    # LAYER 6 -> 7
    video_storage_queue = mp.Queue(maxsize=max_size_video_storage)


    # ============== INSTANTIATE PROCESSES ===================================

    video_reader_process_cfg = {
        "connect_open_timeout_s": fetch_env("VIDEO_STREAM_READER_CONNECTION_OPEN_TIMEOUT_S", VIDEO_STREAM_READER_CONNECTION_OPEN_TIMEOUT_S),
        "connect_retry_delay_s": fetch_env("VIDEO_STREAM_READER_RECONNECT_DELAY", VIDEO_STREAM_READER_RECONNECT_DELAY),
        "connect_max_consecutive_failures": fetch_env("VIDEO_STREAM_READER_MAX_CONSECUTIVE_CONNECTION_FAILURES", VIDEO_STREAM_READER_MAX_CONSECUTIVE_CONNECTION_FAILURES),
        "frame_read_timeout_s": fetch_env("VIDEO_STREAM_READER_FRAME_READ_TIMEOUT_S", VIDEO_STREAM_READER_FRAME_READ_TIMEOUT_S),
        "frame_read_retry_delay_s": fetch_env("VIDEO_STREAM_READER_FRAME_RETRY_DELAY", VIDEO_STREAM_READER_FRAME_RETRY_DELAY),
        "frame_read_max_consecutive_failures": fetch_env("VIDEO_STREAM_READER_FRAME_MAX_CONSECUTIVE_FAILURES", VIDEO_STREAM_READER_FRAME_MAX_CONSECUTIVE_FAILURES),
        "buffer_size": fetch_env("VIDEO_STREAM_READER_BUFFER_SIZE", VIDEO_STREAM_READER_BUFFER_SIZE),
        "expected_aspect_ratio": fetch_env("VIDEO_STREAM_READER_EXPECTED_ASPECT_RATIO", VIDEO_STREAM_READER_EXPECTED_ASPECT_RATIO),
        "processing_shape": fetch_env("VIDEO_STREAM_READER_PROCESSING_SHAPE", VIDEO_STREAM_READER_PROCESSING_SHAPE),
        "queue_out_put_timeout": fetch_env("VIDEO_STREAM_READER_QUEUE_PUT_TIMEOUT", VIDEO_STREAM_READER_QUEUE_PUT_TIMEOUT),
        "poison_pill_timeout": fetch_env("POISON_PILL_TIMEOUT",POISON_PILL_TIMEOUT),
    }

    # Create StreamVideoReader process
    video_reader_process = StreamVideoReader(
        frame_queue=frame_reader_out_queue,
        stop_event=stop_event,
        error_event=error_event,
        video_stream_url=fetch_env("MEDIASERVER_URL_ORIGINAL", None),
        **video_reader_process_cfg,
    )

    # ---------------------------------

    telemetry_reader_process_cfg = {
        "qos_level": fetch_env("MQTT_QOS_LEVEL", MQTT_QOS_LEVEL),
        "max_msg_wait": fetch_env("MQTT_MSG_WAIT_TIMEOUT", MQTT_MSG_WAIT_TIMEOUT),
        "reconnection_delay": fetch_env("MQTT_RECONNECT_DELAY", MQTT_RECONNECT_DELAY),
        "max_incoming_messages": fetch_env("MQTT_MAX_INCOMING_MESSAGES", MQTT_MAX_INCOMING_MESSAGES),
    }

    # Create StreamTelemetryReader process
    telemetry_reader_process = MqttCollectorProcess(
        telemetry_queue=telemetry_reader_out_queue,
        stop_event=stop_event,
        error_event=error_event,
        broker_host=fetch_env("MQTT_HOST", MQTT_HOST),
        broker_port=fetch_env("MQTT_PORT", MQTTS_PORT),
        username=fetch_env("MQTT_USERNAME", None),
        password=fetch_env("MQTT_PASSWORD", None),
        ca_certs_file_path=fetch_env("MQTT_CERT_VALIDATION", None),
        cert_validation=fetch_env("MQTT_CERT_VALIDATION", MQTT_CERT_VALIDATION),
        **telemetry_reader_process_cfg,
    )

    # ---------------------------------

    combiner_process_cfg = {
        "telemetry_buffer_max_size": fetch_env("FRAMETELCOMB_MAX_TELEM_BUFFER_SIZE", FRAMETELCOMB_MAX_TELEM_BUFFER_SIZE),
        "max_time_diff_s": fetch_env("FRAMETELCOMB_MAX_TIME_DIFF", FRAMETELCOMB_MAX_TIME_DIFF),
        "queue_get_timeout": fetch_env("FRAMETELCOMB_QUEUE_GET_TIMEOUT", FRAMETELCOMB_QUEUE_GET_TIMEOUT),
        "queue_put_max_retries": fetch_env("FRAMETELCOMB_QUEUE_PUT_MAX_RETRIES", FRAMETELCOMB_QUEUE_PUT_MAX_RETRIES),
        "queue_put_backoff": fetch_env("FRAMETELCOMB_QUEUE_PUT_BACKOFF", FRAMETELCOMB_QUEUE_PUT_BACKOFF),
        "poison_pill_backoff": fetch_env("POISON_PILL_TIMEOUT", POISON_PILL_TIMEOUT),
    }

    # Create Frame-Telemetry combiner process
    combiner_process = FrameTelemetryCombiner(
        frame_queue=frame_reader_out_queue,
        telemetry_queue=telemetry_reader_out_queue,
        output_queues=models_in_queues,
        error_event=error_event,
        **combiner_process_cfg,
    )

    # ---------------------------------

    detection_process_cfg = {
        "queue_get_timeout": fetch_env("MODELS_QUEUE_GET_TIMEOUT", MODELS_QUEUE_GET_TIMEOUT),
        "queue_put_timeout": fetch_env("MODELS_QUEUE_PUT_TIMEOUT", MODELS_QUEUE_PUT_TIMEOUT),
        "poison_pill_timeout": fetch_env("POISON_PILL_TIMEOUT", POISON_PILL_TIMEOUT),
    }

    # Create DetectionWorker process
    detection_process = DetectionWorker(
        input_queue=detection_in_queue,
        result_queue=detection_results_queue,
        error_event=error_event,
        detection_args=detection_args,
        **detection_process_cfg,

    )

    # ---------------------------------

    segmentation_process_cfg = {
        "queue_get_timeout": fetch_env("MODELS_QUEUE_GET_TIMEOUT", MODELS_QUEUE_GET_TIMEOUT),
        "queue_put_timeout": fetch_env("MODELS_QUEUE_PUT_TIMEOUT", MODELS_QUEUE_PUT_TIMEOUT),
        "poison_pill_timeout": fetch_env("POISON_PILL_TIMEOUT", POISON_PILL_TIMEOUT),
    }

    # Create DetectionWorker process
    segmentation_process = SegmentationWorker(
        input_queue=segmentation_in_queue,
        result_queue=segmentation_results_queue,
        error_event=error_event,
        segmentation_args=segmentation_args,
        **segmentation_process_cfg
    )

    # ---------------------------------

    geo_process_cfg = {
        "queue_get_timeout": fetch_env("MODELS_QUEUE_GET_TIMEOUT", MODELS_QUEUE_GET_TIMEOUT),
        "queue_put_timeout": fetch_env("MODELS_QUEUE_PUT_TIMEOUT", MODELS_QUEUE_PUT_TIMEOUT),
        "poison_pill_timeout": fetch_env("POISON_PILL_TIMEOUT", POISON_PILL_TIMEOUT),
    }

    # Create GeoWorker process
    geo_process = GeoWorker(
        input_queue=geo_in_queue,
        result_queue=geo_results_queue,
        error_event=error_event,
        input_args=input_args,
        drone_args=drone_args,
        **geo_process_cfg,
    )

    # ---------------------------------

    danger_identification_process_cfg = {
        "queue_get_timeout": fetch_env("MODELS_QUEUE_GET_TIMEOUT",MODELS_QUEUE_GET_TIMEOUT),
        "queue_put_timeout": fetch_env("MODELS_QUEUE_PUT_TIMEOUT", MODELS_QUEUE_PUT_TIMEOUT),
        "poison_pill_timeout": fetch_env("POISON_PILL_TIMEOUT", POISON_PILL_TIMEOUT),
    }

    # Create DangerIdentification process
    danger_identification_process = DangerDetectionWorker(
        input_queues=models_results_queues,
        result_queue=danger_detection_result_queue,
        error_event=error_event,
        **danger_identification_process_cfg,
    )

    # ---------------------------------

    annotation_process_cfg = {
        "queue_get_timeout": fetch_env("ANNOTATION_QUEUE_GET_TIMEOUT", ANNOTATION_QUEUE_GET_TIMEOUT),
        "queue_put_timeout": fetch_env("ANNOTATION_QUEUE_PUT_TIMEOUT", ANNOTATION_QUEUE_PUT_TIMEOUT),
        "max_consecutive_failures":  fetch_env("ANNOTATION_MAX_CONSECUTIVE_FAILURES", ANNOTATION_MAX_CONSECUTIVE_FAILURES),
        "max_put_alert_consecutive_failures":  fetch_env("ANNOTATION_MAX_PUT_ALERT_CONSECUTIVE_FAILURES", ANNOTATION_MAX_PUT_ALERT_CONSECUTIVE_FAILURES),
        "max_put_video_consecutive_failures":  fetch_env("ANNOTATION_MAX_PUT_VIDEO_CONSECUTIVE_FAILURES", ANNOTATION_MAX_PUT_VIDEO_CONSECUTIVE_FAILURES),
        "poison_pill_timeout":  fetch_env("POISON_PILL_TIMEOUT", POISON_PILL_TIMEOUT),
    }

    # Create VideoAnnotatorWorker
    annotation_process = AnnotationWorker(
        input_queue=danger_detection_result_queue,
        video_stream_queue=video_stream_queue,
        alerts_stream_queue=notifications_stream_queue,
        error_event=error_event,
        alerts_cooldown_seconds=alerts_cooldown_seconds,
        **annotation_process_cfg,
    )

    # ---------------------------------

    video_writer_process_cfg = {
        "fps": fetch_env("VIDEO_WRITER_FPS", VIDEO_WRITER_FPS),
        "get_frame_timeout": fetch_env("VIDEO_WRITER_GET_FRAME_TIMEOUT", VIDEO_WRITER_GET_FRAME_TIMEOUT),
        # -------- LOCAL SAVE -------------
        "local_video_extension": fetch_env("VIDEO_WRITER_FILE_TYPE_EXTENSION", VIDEO_WRITER_FILE_TYPE_EXTENSION),
        "video_codec": fetch_env("VIDEO_WRITER_CODEC", VIDEO_WRITER_CODEC),
        # -------- STREAM MANAGER -------------
        "stream_manager_queue_max_size": fetch_env("VIDEO_OUT_STREAM_QUEUE_MAX_SIZE", MAX_SIZE_VIDEO_STREAM),
        "stream_manager_queue_get_timeout": fetch_env("VIDEO_OUT_STREAM_QUEUE_GET_TIMEOUT", VIDEO_OUT_STREAM_QUEUE_GET_TIMEOUT),
        "stream_manager_ffmpeg_startup_timeout": fetch_env("VIDEO_OUT_STREAM_FFMPEG_STARTUP_TIMEOUT", VIDEO_OUT_STREAM_FFMPEG_STARTUP_TIMEOUT),
        "stream_manager_ffmpeg_shutdown_timeout": fetch_env("VIDEO_OUT_STREAM_FFMPEG_SHUTDOWN_TIMEOUT", VIDEO_OUT_STREAM_FFMPEG_SHUTDOWN_TIMEOUT),
        "stream_manager_startup_timeout": fetch_env("VIDEO_OUT_STREAM_STARTUP_TIMEOUT", VIDEO_OUT_STREAM_STARTUP_TIMEOUT),
        "stream_manager_shutdown_timeout": fetch_env("VIDEO_OUT_STREAM_SHUTDOWN_TIMEOUT", VIDEO_OUT_STREAM_SHUTDOWN_TIMEOUT),
        # -------- STORAGE_MANAGER ------------
        "storage_manager_handoff_timeout": fetch_env("VIDEO_WRITER_HANDOFF_TIMEOUT", VIDEO_WRITER_HANDOFF_TIMEOUT),
    }

    # Create VideoStreamWriter process
    video_writer_process = VideoProducerProcess(
        input_queue=video_stream_queue,
        output_queue=video_storage_queue,
        error_event=error_event,
        local_video_name="recording",
        media_server_url=fetch_env("MEDIASERVER_URL_ANNOTATED", None),
    )

    # ---------------------------------

    notification_writer_process_cfg = {
        "alerts_get_timeout": fetch_env("ALERTS_GET_TIMEOUT", ALERTS_GET_TIMEOUT),
        "alerts_max_consecutive_failures": fetch_env("ALERTS_MAX_CONSECUTIVE_FAILURES", ALERTS_MAX_CONSECUTIVE_FAILURES),
        "alerts_jpeg_quality": fetch_env("ALERTS_JPEG_COMPRESSION_QUALITY", ALERTS_JPEG_COMPRESSION_QUALITY),
        # ------- WS manager parameters --------
        "ws_manager_ping_interval": fetch_env("WS_MANAGER_PING_INTERVAL", WS_MANAGER_PING_INTERVAL),
        "ws_manager_ping_timeout": fetch_env("WS_MANAGER_PING_TIMEOUT", WS_MANAGER_PING_TIMEOUT),
        "ws_manager_broadcast_timeout": fetch_env("WS_MANAGER_BROADCAST_TIMEOUT", WS_MANAGER_BROADCAST_TIMEOUT),
        "ws_manager_thread_close_timeout": fetch_env("WS_MANAGER_THREAD_CLOSE_TIMEOUT", WS_MANAGER_THREAD_CLOSE_TIMEOUT),
        # ------- DB manager parameters --------
        "db_manager_pool_size": fetch_env("DB_MANAGER_POOL_SIZE", DB_MANAGER_POOL_SIZE),
        "db_manager_max_overflow": fetch_env("DB_MANAGER_MAX_OVERFLOW", DB_MANAGER_MAX_OVERFLOW),
        "db_manager_queue_get_timeout": fetch_env("DB_MANAGER_QUEUE_WAIT_TIMEOUT", DB_MANAGER_QUEUE_WAIT_TIMEOUT),
        "db_manager_thread_close_timeout": fetch_env("DB_MANAGER_THREAD_CLOSE_TIMEOUT", DB_MANAGER_THREAD_CLOSE_TIMEOUT),
        "db_manager_alerts_queue_size": fetch_env("DB_MANAGER_QUEUE_SIZE", DB_MANAGER_QUEUE_SIZE),
    }

    # Create VideoStreamWriter process
    notification_writer_process = NotificationsStreamWriter(
        input_queue=notifications_stream_queue,
        error_event=error_event,
        log_file_path= "alerts.log",
        websocket_host=fetch_env("WS_HOST", None),
        websocket_port=fetch_env("WS_PORT", WSS_PORT),
        database_url=fetch_env("DB_URL", None),
        **notification_writer_process_cfg,
    )

    # ---------------------------------

    video_storage_process_cfg =  {
        "max_retries": fetch_env("VIDEO_OUT_STORE_MAX_UPLOAD_RETRIES", VIDEO_OUT_STORE_MAX_UPLOAD_RETRIES),
        "retry_backoff": fetch_env("VIDEO_OUT_STORE_RETRY_BACKOFF_TIME", VIDEO_OUT_STORE_RETRY_BACKOFF_TIME),
    }

    video_storage_process = VideoPersistenceProcess(
        input_queue=video_storage_queue,
        storage_url=fetch_env("VIDEO_STORAGE_URL", None),
        delete_local_on_success=fetch_env("DELETE_LOCAL_VIDEO", True),
        **video_storage_process_cfg,
    )

    # ============== START PROCESSES (REVERSE ORDER) ===================================

    # LAYER 7
    video_storage_process.start()
    # LAYER 6
    notification_writer_process.start()
    video_writer_process.start()
    # LAYER 5
    annotation_process.start()
    # LAYER 4
    danger_identification_process.start()
    # LAYER 3
    geo_process.start()
    segmentation_process.start()
    detection_process.start()
    # LAYER 2
    combiner_process.start()
    # LAYER 1
    telemetry_reader_process.start()
    video_reader_process.start()

    # ============== JOIN PROCESSES (SEQUENTIAL ORDER) ===================================

    # LAYER 1
    video_reader_process.join()
    telemetry_reader_process.join()
    # LAYER 2
    combiner_process.join()
    # LAYER 3
    detection_process.join()
    segmentation_process.join()
    geo_process.join()
    # LAYER 4
    danger_identification_process.join()
    # LAYER 5
    annotation_process.join()
    # LAYER 6
    video_writer_process.join()
    notification_writer_process.join()
    # LAYER 7
    video_storage_process.join()

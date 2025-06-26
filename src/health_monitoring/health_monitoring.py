from time import time
from typing import Any
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import torch
import cv2
import shutil

from src.health_monitoring.tracking.detection import perform_tracking
from src.health_monitoring.tracking.history import HistoryTracker

from src.health_monitoring.anomaly_detection.anomaly_detection import perform_anomaly_detection_statistical, compute_gsd_from_file, compute_area_fraction
from src.health_monitoring.anomaly_detection.anomaly_detection import merge_previous_anomaly_status_current_detections

from src.health_monitoring.output.frames import annotate_and_save_frame
from src.health_monitoring.output.alerts import send_alert


def perform_health_monitoring_analysis(
        input_args: dict[str:Any],
        output_args: dict[str:Any],
        tracking_args: dict[str:Any],
        anomaly_detection_args: dict[str:Any],
        drone_args: dict[str:Any],
) -> None:

    # TODO:REMOVE
    shutil.rmtree("TMP_HEALTH")
    Path("TMP_HEALTH").mkdir(exist_ok=False)

    # ============== CREATE OUTPUT DIRECTORY ===================================

    # clean (delete and recreate) the output directory if it already exists
    output_dir = Path(output_args["output_dir"])
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # ============== LOAD AI MODELS ===================================

    # Load YOLO tracking model
    detection_model_checkpoint = tracking_args.pop("model_checkpoint")
    tracker = YOLO(detection_model_checkpoint, task="detect")  # Animal detection & tracking model

    anomaly_detection_model_ckpt = anomaly_detection_args.pop("model_checkpoint")
    if anomaly_detection_model_ckpt is None:
        anomaly_detection_model = 'stat'
    else:
        anomaly_detection_model = torch.load(anomaly_detection_model_ckpt)

    # ============== LOAD DETECTION CLASSES INFO ===================================

    # prepare detection classes names and number
    classes_names = tracker.names  # Dictionary of class names
    num_classes = len(classes_names)

    # ============== LOAD FLIGHT INFO ===================================

    # Open drone flight data
    flight_data_file_path = Path(input_args["flight_data"])
    flight_data_file = open(flight_data_file_path, "r")

    # ============== LOAD INPUT VIDEO INFO ===================================

    # Open video and get properties
    cap = cv2.VideoCapture(input_args["source"])
    assert cap.isOpened(), "Error reading video file"

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # compute the aspect ratio of the video frame
    # used to update normalized yolo detections
    aspect_ratio = frame_width / frame_height

    # ============== LOAD ALERTS FILE ===================================

    alerts_file_path = (output_dir / output_args["alert_file_name"]).with_suffix(".txt")
    alerts_file = open(alerts_file_path, "w")

    # ============== LOAD VIDEO WRITER ===================================

    annotated_video_path = (output_dir / output_args["annotated_video_name"]).with_suffix(".mp4")
    annotated_writer = cv2.VideoWriter(
        filename=annotated_video_path,
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
        fps=fps,
        frameSize=(frame_width, frame_height)
    )

    # ============== INITIALIZE HISTORY TRACKER ===================================

    # all timeseries must have equal length, compensate for different sizes when used as-is
    if anomaly_detection_model != 'stat':
        anomaly_detection_args["time_window_size"] += 3

    history_update_period_frames = max(1, int(anomaly_detection_args["history_update_period_seconds"] * fps))
    history_update_period_seconds = history_update_period_frames / fps
    history_tracker = HistoryTracker(
        anomaly_detection_args["time_window_size"],
        history_update_period_frames,
        history_update_period_seconds,
    )

    # ============== SETUP ANOMALY DETECTION CONTINUITY ===================================

    anomaly_detection_period_frames = history_update_period_frames * anomaly_detection_args["anomaly_detection_period"]
    previous_ids = []
    previous_anomaly_status = []
    
    # ============== BEGIN VIDEO PROCESSING ===================================

    # Frame counter
    frame_id = 0

    # Alert cooldown initialization
    alerts_frames_cooldown = max(1, int(input_args["alerts_cooldown_seconds"] * fps))   # convert cooldown from seconds to frames
    last_alert_frame_id = - fps  # to avoid dealing with initial None value, at frame 0 alert is allowed

    # Time keeper
    processing_start_time = time()

    # Video processing loop
    while cap.isOpened():
        iteration_start_time = time()
        success, frame = cap.read()
        if not success or frame_id > 2000:
            print("Video processing has been successfully completed.")
            break

        frame_id += 1  # Update frame ID
        print(f"\n------------- Processing frame {frame_id}/{total_frames}-----------")

        # ============== PERFORM TRACKING ===========================================
        (
            ids_list,
            classes,
            _,
            _,
            scalenorm_boxes_centers,
            boxes_corner1,
            boxes_corner2,
        ) = perform_tracking(
                detector=tracker,
                frame=frame,
                tracking_args=tracking_args,
                aspect_ratio=aspect_ratio,
        )

        with open(output_dir / "tracks.txt", "a") as f:
            f.write(str(sorted(ids_list)))
            f.write("\n")

        # ============== UPDATE HISTORY (based on update frequency) =================

        # Update the history of movements based on tracking results
        positions_list = [] if len(ids_list) == 0 else scalenorm_boxes_centers.tolist()

        # ... once every 'history_update_period_frames' frames
        update_history = ((frame_id-1) % history_update_period_frames == 0)
        if update_history:
            history_tracker.update(ids_list, positions_list)

        # ============== PERFORM ANOMALY DETECTION ===================================

        # Do not perform anomaly detection until history has been filled
        # i.i, skip for the first 'window_size' updates of the history
        # first applied at after at least (history_update_period_frames * window_size) frames
        run_anomaly_detection = (
                ((frame_id-1) % anomaly_detection_period_frames == 0) and
                (frame_id-1) > (history_tracker.window_size * history_update_period_frames) and
                len(ids_list) >= 3  # at least two references to determine third anomaly
        )

        if run_anomaly_detection:
            print("applying AD")
            # Compute ground sampling distance (GSD)
            meters_per_pixel = compute_gsd_from_file(
                drone_args=drone_args,
                flight_data_file=flight_data_file,
                frame_id=frame_id,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            # convert radius from meters into a fraction of the image size using GSD
            radius_r = compute_area_fraction(
                radius_meters=anomaly_detection_args["radius_meters"],
                meters_per_pixel=meters_per_pixel,
                frame_width=frame_width,
            )

            if anomaly_detection_model == 'stat':
                are_anomalous = perform_anomaly_detection_statistical(
                    history=history_tracker,
                    radius_r=radius_r,
                    knn_k=anomaly_detection_args["knn_k"],
                    majority_vote_threshold=anomaly_detection_args["majority_vote_threshold"],
                    current_ids_list=ids_list,
                    frame_id=frame_id,
                )
            else:
                raise NotImplementedError

            # save ids and anomaly status for next iteration when anomaly detection is not executed
            previous_ids = ids_list
            previous_anomaly_status = are_anomalous

        else:
            are_anomalous = merge_previous_anomaly_status_current_detections(
                current_ids=ids_list,
                previous_ids=previous_ids,
                previous_anomaly_status=previous_anomaly_status,
            )

        # ============== RAISE ALERTS IF NEEDED ================

        # verify whether the cooldown has passed
        cooldown_has_passed = (frame_id - last_alert_frame_id) >= alerts_frames_cooldown
        # verify whether anomalies are detected
        anomaly_exists = np.any(are_anomalous)

        # report anomalous behaviour (if needed)
        # alerts only raised when anomaly detection has actually been performed run during the iteration
        raise_alert = cooldown_has_passed and anomaly_exists and run_anomaly_detection
        if raise_alert:
            num_anomalies = np.count_nonzero(are_anomalous)  # count the number of anomalies
            send_alert(alerts_file, frame_id, num_anomalies)    # raise textual alert
            last_alert_frame_id = frame_id  # update for cooldown

        # ============== ANNOTATE FRAME ==================================

        annotate_and_save_frame(
            output_dir=output_dir,
            annotated_writer=annotated_writer,
            frame=frame,
            frame_id=frame_id,
            raise_alert=raise_alert,
            num_classes=num_classes,
            classes_names=classes_names,
            classes=classes,
            are_anomalous=are_anomalous,
            ids=ids_list,
            boxes_corner1=boxes_corner1,
            boxes_corner2=boxes_corner2,
        )

        iteration_time = (time() - iteration_start_time)
        print(f"Iteration completed in {iteration_time*1000:.1f} ms. Equivalent fps = {1/iteration_time:.3f}")

    """ Processing completed, print stats and release resources"""

    total_time = time() - processing_start_time
    processing_rate = total_frames / total_time
    print(f"Processing rate: {processing_rate:.1f} fps. Real time: {processing_rate >= fps}")

    # close files
    alerts_file.close()
    flight_data_file.close()

    # close videos
    cap.release()
    annotated_writer.release()

    print(f"Videos and alerts log have been saved at {output_dir}")

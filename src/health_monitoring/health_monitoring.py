from time import time
from typing import Any
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import torch
import cv2

from tracking.detection import perform_tracking
from tracking.history import HistoryTracker

from anomaly_detection.anomaly_detection import perform_anomaly_detection

from output.frames import annotate_video
from output.alerts import send_alert


def perform_health_monitoring_analysis(
        input_args: dict[str:Any],
        output_args: dict[str:Any],
        tracking_args: dict[str:Any],
        anomaly_detection_args: dict[str:Any],
        drone_args: dict[str:Any],
) -> None:

    # ============== CREATE OUTPUT DIRECTORY ===================================

    output_dir = Path(output_args["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============== LOAD AI MODELS ===================================

    # Load YOLO detector model
    detection_model_checkpoint = tracking_args.pop("model_checkpoint")
    tracker = YOLO(detection_model_checkpoint, task="detect")  # Animal detection & tracking model

    # Load anomaly detection model
    anomaly_detection_model = torch.load(anomaly_detection_args.pop("model_checkpoint"))

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

    # avoids unreasonable video strides
    input_args["vid_stride"] = max(1, min(input_args["vid_stride"], total_frames))
    print(f"Processing 1 frame every {input_args['vid_stride']}")

    # ============== LOAD ALERTS FILE ===================================

    alerts_file_path = (output_dir / output_args["alert_file_name"]).with_suffix(".txt")
    alerts_file = open(alerts_file_path, "w")

    # ============== LOAD VIDEO WRITERS ===================================

    annotated_video_path = (output_dir / output_args["annotated_video_name"]).with_suffix(".mp4")
    annotated_writer = cv2.VideoWriter(
        filename=annotated_video_path,
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
        fps=fps,
        frameSize=(frame_width, frame_height)
    )

    # ============== INITIALIZE HISTORY TRACKER ===================================

    history_tracker = HistoryTracker(input_args["time_window_size"])
    
    # ============== BEGIN VIDEO PROCESSING ===================================

    # Frame counter
    frame_id = 0
    processed_frames_counter = 0

    # Alert cooldown initialization
    alerts_frames_cooldown = input_args["alerts_cooldown_seconds"] * fps   # convert cooldown from seconds to frames
    last_alert_frame_id = - fps  # to avoid dealing with initial None value, at frame 0 alert is allowed

    # Time keeper
    processing_start_time = time()

    # Video processing loop
    while cap.isOpened():
        iteration_start_time = time()
        success, frame = cap.read()
        if not success:
            print("Video processing has been successfully completed.")
            break

        if frame_id % input_args["vid_stride"] != 0:
            frame_id += 1  # Update frame ID
            continue  # go to next frame directly (processes 1 frame every 'vid_stride' frames)

        processed_frames_counter += 1  # update the actual number of frames processed
        frame_id += 1  # Update frame ID
        print(f"\n------------- Processing frame {frame_id}/{total_frames}-----------")

        """ 
        STEP 1:
        Perform object tracking
        """
        # Track animals in frame
        (
            ids_list,
            classes,
            boxes_centers,
            norm_boxes_centers,
            scalenorm_boxes_centers,
            boxes_corner1,
            boxes_corner2,
        ) = perform_tracking(
                detector=tracker,
                frame=frame,
                tracking_args=tracking_args,
                aspect_ratio=aspect_ratio,
        )

        # Update the history of movements based on tracking results
        positions_list = [] if len(ids_list) == 0 else scalenorm_boxes_centers.tolist()
        history_tracker.update(ids_list, positions_list)

        """ 
        STEP2 2:
        Perform anomaly detection
        """

        are_anomalous = perform_anomaly_detection(
            anomaly_detector=anomaly_detection_model,
            anomaly_detection_args=anomaly_detection_args,
            history=history_tracker,
            input_args=input_args,
            drone_args=drone_args,
            flight_data_file=flight_data_file,
            frame_id=frame_id,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        """
        STEP 3:
        Raise alert if anomaly is detected
        """

        # verify whether the cooldown has passed
        cooldown_has_passed = (frame_id - last_alert_frame_id) >= alerts_frames_cooldown
        # verify whether anomalies are detected
        anomaly_exists = np.any(are_anomalous)

        # if cooldown has passed and animal(s) are behaving abnormally, report with the appropriate string
        if cooldown_has_passed and anomaly_exists:
            num_anomalies = np.count_nonzero(are_anomalous)  # count the number of anomalies
            send_alert(alerts_file, frame_id, num_anomalies)    # raise textual alert
            last_alert_frame_id = frame_id  # update for cooldown

        """
        STEP 4: 
        Annotate video
        """
        annotate_video(
            output_dir=output_dir,
            annotated_writer=annotated_writer,
            frame=frame,
            frame_id=frame_id,
            cooldown_has_passed=cooldown_has_passed,
            anomaly_exists=anomaly_exists,
            classes=classes,
            are_anomalous=are_anomalous,
            boxes_corner1=boxes_corner1,
            boxes_corner2=boxes_corner2,
        )

        iteration_time = (time() - iteration_start_time) * 1000
        print(f"Iteration completed in {iteration_time:.1f} ms. Equivalent fps = {1/iteration_time:.1f}")

    """ Processing completed, print stats and release resources"""

    total_time = time() - processing_start_time
    print(f"Danger Analysis for {processed_frames_counter} frames (out of {total_frames}) completed in {total_time:.1f} seconds")
    real_processing_rate = processed_frames_counter / total_time
    print(f"Real processing rate: {real_processing_rate:.1f} fps. Real time: {real_processing_rate >= fps}")
    apparent_processing_rate = total_frames / total_time
    print(f"Apparent processing rate: {apparent_processing_rate:.1f} fps. Real time: {apparent_processing_rate >= fps}")

    # close files
    alerts_file.close()
    flight_data_file.close()

    # close videos
    cap.release()
    annotated_writer.release()

    print(f"Videos and alerts log have been saved at {output_dir}")

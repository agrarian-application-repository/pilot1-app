from pathlib import Path
from time import time
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO
import torch

from utils import *


RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
PURPLE = (128, 0, 128)

CLASS_COLOR = [BLUE, PURPLE]


def perform_health_monitoring_analysis(
        input_args: dict[str:Any],
        output_args: dict[str:Any],
        tracking_args: dict[str:Any],
        anomaly_detection_args: dict[str:Any],
) -> None:

    output_dir = Path(output_args["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============== LOAD AI MODELS ===================================

    # Load YOLO detector model
    detection_model_checkpoint = tracking_args.pop("model_checkpoint")
    tracker = YOLO(detection_model_checkpoint, task="detect")  # Animal detection & tracking model

    anomaly_detector = torch.load(anomaly_detection_args.pop("model_path"))

    # ============== LOAD VIDEO INFO ===================================

    # Open video and get properties
    cap = cv2.VideoCapture(input_args["source"])
    assert cap.isOpened(), "Error reading video file"

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    aspect_ratio = frame_width / frame_height

    # avoids unreasonable video strides
    input_args["vid_stride"] = max(1, min(input_args["vid_stride"], total_frames))

    # ============== LOAD ALERTS FILE ===================================

    alerts_file_path = (output_dir / output_args["alert_file_name"]).with_suffix(".txt")
    alerts_file = open(alerts_file_path, "w")

    # ============== LOAD VIDEO WRITERS ===================================

    annotated_writer = None

    if output_args["save_videos"]:
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
    alerts_frames_cooldown = output_args["alerts_cooldown_seconds"] * fps   # convert cooldown from seconds to frames
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

        """ Perform object tracking"""

        # Track animals in frame
        classes, boxes_centers, normalized_boxes_centers, boxes_corner1, boxes_corner2 = perform_tracking(tracker, history_tracker, frame, tracking_args, aspect_ratio)

        """ Perform anomaly detection"""

        # TODO implement
        are_anomalous = perform_anomaly_detection(anomaly_detector, history_tracker, anomaly_detection_args)
        anomaly_exists = np.any(are_anomalous)

        """ Raise alert if anomaly is detected"""

        # if cooldown has passed and an animal is behaving abnormally, report them with the appropriate string
        cooldown_has_passed = (frame_id - last_alert_frame_id) >= alerts_frames_cooldown
        if cooldown_has_passed and anomaly_exists:
                send_alert(alerts_file, frame_id)
                last_alert_frame_id = frame_id

        """ Annotate video if it has to be saved"""

        # annotations can be skipped if videos are not be saved and no animal is in behaving abnormally
        if output_args["save_videos"] or (anomaly_exists and cooldown_has_passed):

            annotated_frame = frame.copy()  # copy of the original frame on which to draw
            draw_detections(annotated_frame, classes, are_anomalous, boxes_corner1, boxes_corner2)

            if output_args["save_videos"]:  # if the annotation code has been entered because saving the videos ...
                # save the annotated rgb mask and annotated frame
                annotated_writer.write(annotated_frame)
            if anomaly_exists:  # if annotation code has been entered because an animal behaviour is anomalous after cooldown ...
                annotated_img_path = Path(output_dir, f"anomaly_frame_{frame_id}_annotated.png")
                cv2.imwrite(annotated_img_path, annotated_frame)

        iteration_time = (time() - iteration_start_time) * 1000
        print(f"Iteration completed in {iteration_time:.1f} ms. Equivalent fps = {1/iteration_time:.1f}")

    """ Processing completed, print stats and release resources"""

    total_time = time() - processing_start_time
    print(f"Danger Analysis for {processed_frames_counter} frames (out of {total_frames}) completed in {total_time:.1f} seconds")
    real_processing_rate = processed_frames_counter / total_time
    print(f"Real processing rate: {real_processing_rate:.1f} fps. Real time: {real_processing_rate >= fps}")
    apparent_processing_rate = total_frames / total_time
    print(f"Apparent processing rate: {apparent_processing_rate:.1f} fps. Real time: {apparent_processing_rate >= fps}")

    alerts_file.close()

    cap.release()

    if annotated_writer is not None:
        annotated_writer.release()

    print(f"Videos and alerts log have been saved at {output_dir}")

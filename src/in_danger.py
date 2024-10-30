from typing import Any
import cv2
import numpy as np
import torch
from ultralytics import YOLO


import cv2
import numpy as np
from ultralytics import YOLO

def perform_in_danger_analysis(video_path="path/to/your/video.mp4",
                               detector_model="yolo-animal-detection-model.pt",
                               segmenter_model="yolo-dangerous-terrain-segmentation.pt",
                               output_alert_file="alerts.txt",
                               annotated_video_path="in_danger_annotated.avi",
                               mask_video_path="in_danger_masks.avi") -> None:
    # Load YOLO models
    detector = YOLO(detector_model)  # Animal detection model
    segmenter = YOLO(segmenter_model)  # Dangerous terrain segmentation model

    # Output file for alerts
    output_file = open(output_alert_file, "w")

    # Open video and get properties
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    width, height, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Video writers for annotated and mask videos
    annotated_writer = cv2.VideoWriter(annotated_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    mask_writer = cv2.VideoWriter(mask_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Frame counter
    frame_id = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Video processing has been successfully completed.")
            break

        frame_id += 1  # Update frame ID for logging

        # Detect animals in frame
        detection_results = detector.predict(frame)
        segment_results = segmenter.predict(frame)

        # Parse detection results to get bounding boxes
        detections = detection_results[0].boxes.cpu().numpy()  # Assuming results are in a .boxes format
        dangerous_areas = segment_results[0].masks.cpu().numpy()  # Assuming masks are in .masks format

        # Initialize mask for dangerous areas
        dangerous_mask = np.zeros((height, width), dtype=np.uint8)
        for mask in dangerous_areas:
            dangerous_mask = np.maximum(dangerous_mask, mask)  # Stack binary masks for all dangerous areas

        # Overlay mask for dangerous areas (to be drawn in red in annotated video)
        dangerous_area_overlay = cv2.cvtColor(dangerous_mask * 255, cv2.COLOR_GRAY2BGR)
        dangerous_area_overlay[:, :, 1:] = 0  # Red color channel only

        # Initialize frames for annotated and mask videos
        annotated_frame = frame.copy()
        mask_frame = np.zeros((height, width, 3), dtype=np.uint8)  # Background in black

        # Process each detection to draw bounding boxes and safety ellipses
        for box in detections:
            # Bounding box details
            x, y, w, h = int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])

            # Draw bounding box on annotated frame (blue)
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.rectangle(mask_frame, (x, y), (x + w, y + h), (255, 0, 0), -1)  # Filled blue on mask frame

            # Generate safety ellipse around the bounding box
            center_coordinates = (x + w // 2, y + h // 2)
            axes_length = (int(2.5 * w), int(2.5 * h))  # Ellipse size scaled up by 2.5
            cv2.ellipse(annotated_frame, center_coordinates, axes_length, 0, 0, 360, (0, 255, 0), 2)  # Green in annotated
            cv2.ellipse(mask_frame, center_coordinates, axes_length, 0, 0, 360, (0, 255, 0), -1)  # Filled green in mask

            # Create a temporary mask for the safety area ellipse and check intersections
            safety_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.ellipse(safety_mask, center_coordinates, axes_length, 0, 0, 360, 255, -1)

            # Check for intersection between safety area and dangerous areas
        intersection = cv2.bitwise_and(safety_mask, dangerous_mask)
        if np.any(intersection > 0):  # Non-zero intersection indicates overlap
            # Write alert to file
            output_file.write(f"Alert: Frame {frame_id}, Animal(s) near or in dangerous area\n")
            # Highlight intersection area in yellow on the mask frame
            mask_frame[np.where(intersection > 0)] = (0, 255, 255)  # Yellow for intersection

        # Overlay dangerous areas in red on the annotated frame
        annotated_frame = cv2.addWeighted(annotated_frame, 1, dangerous_area_overlay, 0.5, 0)

        # Dangerous areas in red on mask frame
        mask_frame[np.where(dangerous_mask > 0)] = (0, 0, 255)  # Red for dangerous areas

        # Write frames to video writers
        annotated_writer.write(annotated_frame)
        mask_writer.write(mask_frame)

    # Release resources
    cap.release()
    annotated_writer.release()
    mask_writer.release()
    output_file.close()
    print("Processing complete. Videos and alert log have been saved.")



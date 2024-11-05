import os
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO, solutions


def perform_in_danger_analysis(
    video_path="data/sheep_videos/23.11.23-16.MP4",
    detector_model="yolo11m.pt",
    segmenter_model="yolo11m.pt",
    base_path="experiments/maich_in_danger",
    output_alert_file="alerts.txt",
    annotated_video_path="in_danger_annotated.mp4",
    mask_video_path="in_danger_masks.mp4",
    heatmap_video_path="in_danger_heatmap.mp4",
) -> None:
    output_alert_file = os.path.join(base_path, output_alert_file)
    annotated_video_path = os.path.join(base_path, annotated_video_path)
    mask_video_path = os.path.join(base_path, mask_video_path)
    heatmap_video_path = os.path.join(base_path, heatmap_video_path)

    classes = None
    if detector_model in [
        "yolo11n.pt",
        "yolo11s.pt",
        "yolo11m.pt",
        "yolo11l.pt",
        "yolo11x.pt",
    ]:
        classes = [18]

    # Load YOLO models
    detector = YOLO(detector_model, task="detect")  # Animal detection model
    # segmenter = YOLO(segmenter_model, task="segment")  # Dangerous terrain segmentation model

    # Output file for alerts
    os.makedirs(base_path, exist_ok=True)
    output_file = open(output_alert_file, "w")

    # Open video and get properties
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    width, height, fps = (
        int(cap.get(x))
        for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
    )

    heatmap_solution = solutions.Heatmap(
        model=detector_model,
        classes=classes,
        show=False,
        show_in=False,
        show_out=False,
    )

    annotated_writer = cv2.VideoWriter(
        annotated_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    mask_writer = cv2.VideoWriter(
        mask_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    heatmap_writer = cv2.VideoWriter(
        heatmap_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    # Frame counter
    frame_id = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Video processing has been successfully completed.")
            break

        frame_id += 1  # Update frame ID for logging

        # Detect animals in frame
        detection_results = detector.predict(frame, classes=classes)
        # segment_results = segmenter.predict(frame)    # todo unlock

        # Parse detection results to get bounding boxes
        xywh_boxes = detection_results[0].boxes.cpu().xywh.numpy()
        xyxy_boxes = detection_results[0].boxes.cpu().xyxy.numpy()
        num_detections = xywh_boxes.shape[0]

        # todo unlock
        # dangerous_areas = (
        #    segment_results[0].masks.cpu().numpy()
        # )  # Assuming masks are in .masks format

        # todo unlock
        # Initialize mask for dangerous areas
        # dangerous_mask = np.zeros(height, width), dtype=np.uint8)
        # for mask in dangerous_areas:
        #    dangerous_mask = np.maximum(
        #        dangerous_mask, mask
        #    )  # Stack binary masks for all dangerous areas

        # todo lock
        dangerous_mask = np.zeros((height, width), dtype=np.uint8)
        dangerous_mask[300:500, 100 : width - 100] = 1

        # Initialize frames for annotated and mask videos
        annotated_frame = frame.copy()  # copy of the original frame on which to draw
        rgb_mask_frame = np.zeros(
            (height, width, 3), dtype=np.uint8
        )  # empty rgb mask to color

        # Process each detection to draw bounding boxes and safety circles
        for idx in range(num_detections):
            x, y, w, h = xywh_boxes[idx, :].astype(int)

            # Generate safety circle around the bounding box
            center_coordinates = (x, y)
            radius = int(1.5 * max(w, h))

            # Draw safety circle on annotated frame around bounding box (green)
            cv2.circle(annotated_frame, center_coordinates, radius, (0, 255, 0), 2)
            # Draw safety circle on rgb mask around bounding box (green - fill)
            cv2.circle(rgb_mask_frame, center_coordinates, radius, (0, 255, 0), -1)

        # Create a safety mask from the green channel (circles) of the rgb_mask frame
        safety_mask = np.copy(rgb_mask_frame[:, :, 1] // 255).astype(np.uint8)

        for idx in range(num_detections):
            x1, y1, x2, y2 = xyxy_boxes[idx, :].astype(int)
            # Draw bounding box on annotated frame (blue), on top of safety circles
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # Draw bounding box on rgb mask frame (blue - fill), on top of the safety circles
            cv2.rectangle(rgb_mask_frame, (x1, y1), (x2, y2), (255, 0, 0), -1)

        # Overlay dangerous areas in red on the annotated frame
        alpha = 0.5
        red_overlay = np.zeros_like(annotated_frame)
        red_overlay[dangerous_mask == 1] = [0, 0, 255]  # Red color channel only
        annotated_frame = cv2.addWeighted(
            red_overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame
        )

        # Dangerous areas in red on mask frame
        rgb_mask_frame[np.where(dangerous_mask > 0)] = (0, 0, 255)

        # Check for intersection between safety area and dangerous areas
        intersection = np.logical_and(safety_mask, dangerous_mask)

        if np.any(intersection > 0):  # Non-zero intersection indicates overlap
            # Write alert to file
            output_file.write(
                f"Alert: Frame {frame_id} - Animal(s) near or in dangerous area\n"
            )
            # Highlight intersection area in yellow on the mask frame
            rgb_mask_frame[np.where(intersection > 0)] = (0, 255, 255)

        # heatmap
        heatmap_frame = heatmap_solution.generate_heatmap(frame)
        # print("heatmap frame shape: ", heatmap_frame.shape)

        # Overlay the text on the image
        text = f"N. Detected: {num_detections}"
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_color = (0, 0, 0)
        fill_color = (255, 255, 255)
        thickness = 1
        line_type = cv2.LINE_AA
        org = (10, height - 10)  # text position in image

        (text_width, text_height), _ = cv2.getTextSize(
            text=text, fontFace=font_face, fontScale=font_scale, thickness=thickness
        )
        textbox_coord_ul = (org[0] - 5, org[1] - text_height - 5)
        textbox_coord_br = (org[0] + text_width + 5, org[1] + 5)

        # Draw white rectangle as background
        cv2.rectangle(
            annotated_frame, textbox_coord_ul, textbox_coord_br, fill_color, cv2.FILLED
        )

        annotated_frame = cv2.putText(
            img=annotated_frame,
            text=text,
            org=org,
            fontFace=font_face,
            fontScale=font_scale,
            color=text_color,
            thickness=thickness,
            lineType=line_type,
        )

        # cv2.imwrite(os.path.join(base_path, "safety_mask.png"), safety_mask * 255)
        # cv2.imwrite(os.path.join(base_path, "dangerous_mask.png"), dangerous_mask * 255)
        # cv2.imwrite(os.path.join(base_path, "annotated_frame.png"), annotated_frame)
        # cv2.imwrite(os.path.join(base_path, "rgb_mask_frame.png"), rgb_mask_frame)
        # cv2.imwrite(os.path.join(base_path, "heatmap.png"), heatmap_frame)

        # Write frames to video writers
        annotated_writer.write(annotated_frame)
        mask_writer.write(rgb_mask_frame)
        heatmap_writer.write(heatmap_frame)

    # Release resources
    cap.release()
    annotated_writer.release()
    mask_writer.release()
    output_file.close()
    print(f"Processing complete. Videos and alert log have been saved at {base_path}")


if __name__ == "__main__":
    perform_in_danger_analysis()

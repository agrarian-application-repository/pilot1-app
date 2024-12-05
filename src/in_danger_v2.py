from pathlib import Path
from time import time
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO
# from skimage import transform as skt


RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (255, 0, 0)
PURPLE = (128, 0, 128)
CLASS_COLOR = [BLUE, PURPLE]


def perform_in_danger_analysis(
        input_args: dict[str:Any],
        output_args: dict[str:Any],
        detection_args: dict[str:Any],
        segmentation_args: dict[str:Any],
) -> None:

    detection_model_checkpoint = detection_args.pop("model_checkpoint")
    segmentation_model_checkpoint = segmentation_args.pop("model_checkpoint")

    # Load YOLO models
    detector = YOLO(detection_model_checkpoint, task="detect")  # Animal detection model
    segmenter = YOLO(segmentation_model_checkpoint, task="segment")  # Dangerous terrain segmentation model

    # Open video and get properties
    cap = cv2.VideoCapture(input_args["source"])
    assert cap.isOpened(), "Error reading video file"

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir = Path(output_args["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    alerts_file_path = (output_dir / output_args["alert_file_name"]).with_suffix(".txt")
    alerts_file = open(alerts_file_path, "w")

    annotated_writer = None
    mask_writer = None

    if output_args["save_videos"]:

        annotated_video_path = (output_dir / output_args["annotated_video_name"]).with_suffix(".mp4")
        annotated_writer = cv2.VideoWriter(
            filename=annotated_video_path,
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=fps,
            frameSize=(frame_width, frame_height)
        )

        mask_video_path = (output_dir / output_args["mask_video_name"]).with_suffix(".mp4")
        mask_writer = cv2.VideoWriter(
            filename=mask_video_path,
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=fps,
            frameSize=(frame_width, frame_height)
        )

    # Frame counter
    true_frame_id = 0
    frame_id = 0

    # Alert cooldown initialization
    alerts_frames_cooldown = output_args["annotated_video_name"] * fps
    last_alert_frame = - fps    # to avoid dealing with initial None value, at frame 0 alert is allowed

    # Time keeper
    start_time = time()

    # Video processing loop
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Video processing has been successfully completed.")
            break

        true_frame_id += 1  # Update frame ID for logging
        if true_frame_id % input_args["vid_stride"] != 1:
            continue    # skip vid_stride frames

        print(f"\n------------- Processing frame {true_frame_id}/{total_frames}-----------")
        frame_id += 1   # update the actual number of frames processed

        """ Perform detection and get bounding boxes"""

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect animals in frame
        detection_results = detector.predict(source=frame, **detection_args)

        # Parse detection results to get bounding boxes
        classes = detection_results[0].boxes.cls.cpu().numpy().astype(int)
        xywh_boxes = detection_results[0].boxes.xywh.cpu().numpy().astype(int)
        xyxy_boxes = detection_results[0].boxes.xyxy.cpu().numpy().astype(int)

        # Create additional variables to store useful info from the detections
        boxes_centers = xywh_boxes[:, :2]
        boxes_wh = xywh_boxes[:, 2:]
        boxes_corner1 = xyxy_boxes[:, :2]
        boxes_corner2 = xyxy_boxes[:, 2:]
        safety_radiuses = (1.5 * np.max(boxes_wh, axis=0)).astype(int)

        """ Perform segmentation and build dangerous mask"""

        segment_results = segmenter.predict(source=frame, **segmentation_args)

        if segment_results[0].masks is not None:    # danger found in the frame
            masks = segment_results[0].masks.data.int().cpu().numpy()
            dangerous_mask = np.any(masks, axis=0).astype(np.uint8)
            dangerous_mask = cv2.resize(dangerous_mask, dsize=(frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
            # dangerous_mask = skt.resize(dangerous_mask, (frame_height, frame_width), order=0, anti_aliasing=False)

        else:   # frame not found in frame
            dangerous_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

        """ Compute safety areas around each animal, and check for intersections with dangerous areas"""

        # create the safety mask
        safety_mask = create_safety_mask(frame_height, frame_width, boxes_centers, safety_radiuses)

        # create the intersection mask between safety areas and dangerous areas
        intersection = np.logical_and(safety_mask, dangerous_mask)

        # Check for intersection between safety area and dangerous areas, non-zero intersection indicates overlap,
        danger_exists = np.any(intersection > 0)
        cooldown_has_passed = (true_frame_id - last_alert_frame) >= alerts_frames_cooldown

        # if overlap exists and cooldown has passed, send alert and update cooldown period
        if danger_exists and cooldown_has_passed:
            send_alert(alerts_file, true_frame_id)
            last_alert_frame = true_frame_id

        """ Additional annotations if videos are to be saved, or for frames where danger exist"""

        # annotations can be skipped if videos are not be saved and no animal is in danger
        if output_args["save_videos"] or (danger_exists and cooldown_has_passed):

            annotated_frame = frame.copy()  # copy of the original frame on which to draw
            rgb_mask_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

            # draw safety circles
            draw_safety_areas(annotated_frame, rgb_mask_frame, boxes_centers, safety_radiuses)

            # Overlay dangerous areas in red on the annotated frame
            annotated_frame = draw_dangerous_area(annotated_frame, rgb_mask_frame, dangerous_mask, intersection)

            # Highlight intersection area in yellow
            draw_animal_in_danger_area(rgb_mask_frame, intersection)

            # draw detection boxes
            draw_detections(annotated_frame, rgb_mask_frame, classes, boxes_centers, boxes_corner1, boxes_corner2)

            # draw animal count
            draw_count(classes, annotated_frame, frame_height)

            if output_args["save_videos"]:  # if the annotation code has been entered because saving the videos ...
                # save the annotated rgb mask and annotated frame
                mask_writer.write(rgb_mask_frame)
                annotated_writer.write(annotated_frame)

            if danger_exists:   # if annotation code has been entered because an animal is in danger after cooldown ...
                mask_img_path = Path(output_dir, f"danger_frame_{true_frame_id}_mask.png")
                annotated_img_path = Path(output_dir, f"danger_frame_{true_frame_id}_annotated.png")
                cv2.imwrite(mask_img_path, rgb_mask_frame)
                cv2.imwrite(annotated_img_path, annotated_frame)

    total_time = time() - start_time
    processing_speed = total_frames / total_time
    print(f"Detection and segmentation for  {total_frames} completed in end {total_time:.1f} seconds")
    print(f"Processing rate: {processing_speed:.2f} fps")
    print(f"Input video fps: {fps}")
    print(f"Processing is real time: {processing_speed >= fps}")

    """Release resources"""

    alerts_file.close()

    cap.release()

    if annotated_writer is not None:
        annotated_writer.release()
    if mask_writer is not None:
        mask_writer.release()

    print(f"Videos and alerts log have been saved at {output_dir}")


def send_alert(alerts_file, frame_id: int):
    # Write alert to file
    alerts_file.write(f"Alert: True Frame {frame_id} - Animal(s) near or in dangerous area\n")


def create_safety_mask(frame_height, frame_width, boxes_centers, safety_radiuses):
    # Create a coordinate grid for the image
    y, x = np.ogrid[:frame_height, :frame_width]

    # Initialize the mask
    safety_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

    # Process all circles in a vectorized manner
    for box_center, safety_radius in zip(boxes_centers, safety_radiuses):
        # Calculate distance from the center of the circle
        distance_from_center = (x - box_center[0]) ** 2 + (y - box_center[1]) ** 2
        # Add to the mask where the distance is within the radius
        safety_mask[distance_from_center <= safety_radius ** 2] = 1

    return safety_mask


def draw_safety_areas(
    annotated_frame,
    rgb_mask_frame,
    boxes_centers,
    safety_radiuses,
):
    # drawing safety circles & detection boxes
    for (box_center, safety_radius) in zip(boxes_centers, safety_radiuses):
        # Draw safety circle on annotated frame (green)
        cv2.circle(annotated_frame, box_center, safety_radius, GREEN, 2)
        # Draw safety circle on rgb mask around bounding box (green - fill)
        cv2.circle(rgb_mask_frame, box_center, safety_radius, GREEN, cv2.FILLED)


def draw_dangerous_area(
    annotated_frame,
    rgb_mask_frame,
    dangerous_mask,
    intersection
):
    alpha = 0.5
    red_overlay = np.zeros_like(annotated_frame)
    red_overlay[dangerous_mask == 1] = RED  # Red color channel only
    annotated_frame = cv2.addWeighted(red_overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)

    # Dangerous areas in red on mask frame
    rgb_mask_frame[np.where((dangerous_mask - intersection) > 0)] = RED

    return annotated_frame


def draw_animal_in_danger_area(rgb_mask_frame, intersection):
    rgb_mask_frame[np.where(intersection > 0)] = YELLOW


def draw_detections(
    annotated_frame,
    rgb_mask_frame,
    classes,
    boxes_centers,
    boxes_corner1,
    boxes_corner2,
):
    # drawing safety circles & detection boxes
    for obj_class, box_center, box_corner1, box_corner2 in zip(classes, boxes_centers, boxes_corner1, boxes_corner2):
        # Draw bounding box on annotated frame (blue sheep, purple goat), on top of safety circles
        cv2.rectangle(annotated_frame, box_corner1, box_corner2, CLASS_COLOR[obj_class], 2)
        # Draw bounding box on rgb mask frame (fill / blue sheep, purple goat), on top of the safety circles
        cv2.rectangle(rgb_mask_frame, box_corner1, box_corner2, CLASS_COLOR[obj_class], cv2.FILLED)


def draw_count(
    classes,
    annotated_frame,
    frame_height,
):
    # Overlay the count text on the annotated frame
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_color = BLACK
    fill_color = WHITE
    thickness = 1
    line_type = cv2.LINE_AA
    org = (10, frame_height - 10)  # text position in image

    num_classes = classes.max() + 1
    class_counts = np.zeros(num_classes, dtype=np.int32)
    class_counts[: len(np.bincount(classes))] = np.bincount(classes)

    text = ""
    for idx, count in enumerate(class_counts):
        text = f"N. Detected class {idx}: {count}\n"

    (text_width, text_height), _ = cv2.getTextSize(
        text=text,
        fontFace=font_face,
        fontScale=font_scale,
        thickness=thickness
    )
    textbox_coord_ul = (org[0] - 5, org[1] - text_height - 5)
    textbox_coord_br = (org[0] + text_width + 5, org[1] + 5)

    # Draw white rectangle as background
    cv2.rectangle(annotated_frame, textbox_coord_ul, textbox_coord_br, fill_color, cv2.FILLED)

    cv2.putText(
        img=annotated_frame,
        text=text,
        org=org,
        fontFace=font_face,
        fontScale=font_scale,
        color=text_color,
        thickness=thickness,
        lineType=line_type,
    )

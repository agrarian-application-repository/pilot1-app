from collections import defaultdict
from pathlib import Path
from time import time
from typing import Any

import cv2
import numpy as np
from skimage import transform as skt
from ultralytics import YOLO, solutions


RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (255, 0, 0)
GRAY = (230, 230, 230)
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

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    input_args["vid_stride"] = min(input_args["vid_stride"], total_frames)

    output_dir = Path(output_args["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    alerts_file_path = (output_dir / output_args["alert_file_name"]).with_suffix(".txt")
    alerts_file = open(alerts_file_path, "w")
    drone_movement_compensation_file = open(output_dir / "compensation.txt", "w")

    heatmap_solution = None

    annotated_writer = None
    mask_writer = None
    heatmap_writer = None

    if output_args["save_videos"]:

        annotated_video_path = (output_dir / output_args["annotated_video_name"]).with_suffix(".mp4")
        annotated_writer = cv2.VideoWriter(
            filename=annotated_video_path,
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=fps,
            frameSize=(width, height)
        )

        mask_video_path = (output_dir / output_args["mask_video_name"]).with_suffix(".mp4")
        mask_writer = cv2.VideoWriter(
            filename=mask_video_path,
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=fps,
            frameSize=(width, height)
        )

        if output_args["draw_heatmap"]:
            heatmap_video_path = (output_dir / output_args["heatmap_video_name"]).with_suffix(".mp4")
            heatmap_writer = cv2.VideoWriter(
                filename=heatmap_video_path,
                fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                fps=fps,
                frameSize=(width, height)
            )
            heatmap_solution = solutions.Heatmap(
                model=detection_model_checkpoint,
                classes=detection_args["classes"],
                colormap=cv2.COLORMAP_INFERNO,
                show=False,
                show_in=False,
                show_out=False,
            )

    # Store the track history
    track_history = {}

    # Frame counter
    previous_frame = None
    true_frame_id = 0
    frame_id = 0
    start_time = time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Video processing has been successfully completed.")
            break

        true_frame_id += 1  # Update frame ID for logging
        if true_frame_id % input_args["vid_stride"] != 1:
            continue    # skip vid_stride frames

        print(f"\n------------- Processing frame {true_frame_id}/{total_frames}-----------")

        frame_id += 1

        # if frame_id < 2400 or frame_id > 2500:
        #    continue

        """ Perform detection and get bounding boxes"""

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect animals in frame
        tracking_results = detector.track(source=frame, persist=True, **detection_args)

        # Parse detection results to get bounding boxes
        classes = tracking_results[0].boxes.cls.cpu().numpy()
        xywh_boxes = tracking_results[0].boxes.xywh.cpu().numpy().astype(int)
        xyxy_boxes = tracking_results[0].boxes.xyxy.cpu().numpy().astype(int)

        # Parse tracking ID
        if tracking_results[0].boxes.id is not None:
            track_ids = tracking_results[0].boxes.id.int().cpu().tolist()
        else:
            track_ids = None

        # Compute useful data from detections
        num_detections = xywh_boxes.shape[0]
        boxes_centers = xywh_boxes[:, :2]
        boxes_wh = xywh_boxes[:, 2:]
        boxes_corner1 = xyxy_boxes[:, :2]
        boxes_corner2 = xyxy_boxes[:, 2:]
        safety_radiuses = (1.5 * np.max(boxes_wh, axis=0)).astype(int)

        """ Perform segmentation and build dangerous-mask"""

        segment_results = segmenter.predict(source=frame, **segmentation_args)

        if segment_results[0].masks is not None:
            masks = segment_results[0].masks.data.int().cpu().numpy()
            dangerous_mask = np.any(masks, axis=0).astype(np.uint8)
            dangerous_mask = skt.resize(dangerous_mask, (height, width), order=0, anti_aliasing=False)
            # dangerous_mask = cv2.resize(dangerous_mask, dsize=(width, height), interpolation=cv2.INTER_NEAREST)
        else:
            dangerous_mask = np.zeros((height, width), dtype=np.uint8)

        """ Draw safety areas on rgb-mask to determine safety-danger intersection areas"""

        # Initialize frames for annotated and mask videos
        rgb_mask_frame = np.zeros((height, width, 3), dtype=np.uint8)  # empty rgb mask to color

        # Process each detection to draw bounding boxes and safety circles
        for box_center, safety_radius in zip(boxes_centers, safety_radiuses):
            # Draw safety circle on rgb mask around bounding box (green - fill)
            cv2.circle(rgb_mask_frame, box_center, safety_radius, GREEN, cv2.FILLED)

        # Create a safety mask from the green channel (circles) of the rgb_mask frame. shape = (H;W;C)
        safety_mask = np.copy(rgb_mask_frame[:, :, 1] // 255).astype(np.uint8)

        # Check for intersection between safety area and dangerous areas
        intersection = np.logical_and(safety_mask, dangerous_mask)
        if np.any(intersection > 0):  # Non-zero intersection indicates overlap
            # Write alert to file
            alerts_file.write(f"Alert: Frame {frame_id} - Animal(s) near or in dangerous area\n")
            # Highlight intersection area in yellow on the mask frame if video to be saved, skip otherwise
            if output_args["save_videos"]:
                rgb_mask_frame[np.where(intersection > 0)] = YELLOW

        """ Additional annotations if videos are to be saved"""

        # This other annotations can be skipped if videos are not saved
        if output_args["save_videos"]:

            annotated_frame = frame.copy()  # copy of the original frame on which to draw

            # drawing safety circles & detection boxes
            draw_detections_and_safety_areas(
                annotated_frame,
                rgb_mask_frame,
                classes,
                boxes_centers,
                safety_radiuses,
                boxes_corner1,
                boxes_corner2,
            )

            # Overlay dangerous areas in red on the annotated frame
            annotated_frame = draw_dangerous_area(
                annotated_frame,
                rgb_mask_frame,
                dangerous_mask,
                intersection
            )

            if output_args["draw_count"]:
                draw_count(num_detections, annotated_frame, height)

            if output_args["draw_tracks"]:
                draw_tracks(
                    frame,
                    previous_frame,
                    annotated_frame,
                    rgb_mask_frame,
                    boxes_centers,
                    track_history,
                    track_ids,
                    drone_movement_compensation_file,
                )
                # Update previous_frame within the loop for the next iteration
                previous_frame = frame.copy()

            # save the annotated rgb and annotated frame
            mask_writer.write(rgb_mask_frame)
            annotated_writer.write(annotated_frame)

            # create and save the heatmap
            if output_args["draw_heatmap"]:     # todo invarianza movimento drone
                heatmap_frame = heatmap_solution.generate_heatmap(frame)
                heatmap_writer.write(heatmap_frame)

    total_time = time() - start_time
    processing_speed = total_frames / total_time
    print(f"Detection and segmentation for  {total_frames} completed in end {total_time:.1f} seconds")
    print(f"Processing rate: {processing_speed:.2f} fps")
    print(f"Input video fps: {fps}")
    print(f"Processing is real time: {processing_speed >= fps}")

    """Release resources"""

    alerts_file.close()
    drone_movement_compensation_file.close()

    cap.release()

    if annotated_writer is not None:
        annotated_writer.release()
    if mask_writer is not None:
        mask_writer.release()
    if heatmap_writer is not None:
        heatmap_writer.release()

    print(f"Videos and alerts log have been saved at {output_dir}")


def draw_detections_and_safety_areas(
    annotated_frame,
    rgb_mask_frame,
    classes,
    boxes_centers,
    safety_radiuses,
    boxes_corner1,
    boxes_corner2,
):
    # drawing safety circles & detection boxes
    for obj_class, box_center, safety_radius, box_corner1, box_corner2 \
            in zip(classes, boxes_centers, safety_radiuses, boxes_corner1, boxes_corner2):
        #  Draw safety circle on annotated frame (green)
        cv2.circle(annotated_frame, box_center, safety_radius, GREEN, 2)
        # Draw bounding box on annotated frame (blue sheep, purple goat), on top of safety circles
        cv2.rectangle(annotated_frame, box_corner1, box_corner2, CLASS_COLOR[obj_class], 2)
        # Draw bounding box on rgb mask frame (fill / blue sheep, purple goat), on top of the safety circles
        cv2.rectangle(rgb_mask_frame, box_corner1, box_corner2, CLASS_COLOR[obj_class], cv2.FILLED)


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


def draw_count(
    num_detections,
    annotated_frame,
    frame_height,
):
    # Overlay the count text on the annotated frame
    text = f"N. Detected: {num_detections}"
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_color = BLACK
    fill_color = WHITE
    thickness = 1
    line_type = cv2.LINE_AA
    org = (10, frame_height - 10)  # text position in image

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


def compensate_tracks_history_for_drone_movement(
    frame,
    previous_frame,
    track_history,
    drone_movement_compensation_file,

):

    # Initialize ORB detector
    orb = cv2.ORB_create()

    prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Detect keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)

    # Match features using Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Estimate the affine transformation matrix from matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

    if M is not None:  # Ensure M was successfully computed
        # Decompose the matrix for translation and rotation
        tx, ty = M[0, 2], M[1, 2]
        a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]

        # Write shift to file for debugging purposes
        drone_movement_compensation_file.write(f"tx: {tx:.2f}, ty: {ty:.2f}\n")

        # Apply the affine transformation to each track's (x, y) positions in track_history
        for k in track_history.keys():
            track_history[k] = [(a * track_x + b * track_y + tx, c * track_x + d * track_y + ty)
                                for (track_x, track_y) in track_history[k]]
            """
            for i, (track_x, track_y) in enumerate(track_history[k]):
                # Apply affine transformation
                corrected_x = a * track_x + b * track_y + tx
                corrected_y = c * track_x + d * track_y + ty
                # Update the track history with corrected coordinates
                track_history[k][i] = (corrected_x, corrected_y)
            """

    # old:
    """
    prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Calculate shift using phase correlation
    # alternatives: Template Matching and  Block-Based Optical Flow (Pyramidal Lucas-Kanade)
    shift, _ = cv2.phaseCorrelate(np.float32(prev_gray), np.float32(curr_gray))
    shift_x, shift_y = shift  # Displacement in x and y directions

    # Optional: Check if the shift is reasonable
    max_shift = 50
    if abs(shift_x) > max_shift or abs(shift_y) > max_shift:
        drone_movement_compensation_file.write(
            "Warning: Unusually large shift detected. Resetting shift to zero below."
        )
        shift_x, shift_y = 0, 0

    drone_movement_compensation_file.write(f"shift_x: {shift_x:.2f} - shift_y:{shift_y:.2f}\n")
    # update tracks with drone shift
    for k in track_history.keys():
        for i, (track_x, track_y) in enumerate(track_history[k]):
            track_history[k][i] = (track_x + shift_x, track_y + shift_y)
    """


def draw_tracks(
    frame,
    previous_frame,
    annotated_frame,
    rgb_mask_frame,
    boxes_centers,
    track_history,
    track_ids,
    drone_movement_compensation_file
):
    # Only perform compensation steps if there's a previous frame, before appending the current tracking points
    if previous_frame is not None:
        compensate_tracks_history_for_drone_movement(
            frame,
            previous_frame,
            track_history,
            drone_movement_compensation_file
        )

    if track_ids is not None:
        for (x, y), track_id in zip(boxes_centers, track_ids):
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=GRAY, thickness=5)
            cv2.polylines(rgb_mask_frame, [points], isClosed=False, color=GRAY, thickness=5)

    for track_id in track_history.keys():
        track = track_history[track_id]
        if len(track) > 100:  # retain 100 frames of track
            track.pop(0)


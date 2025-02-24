import cv2
from pathlib import Path


RED = (255, 0, 0)
BLUE = (0, 0, 255)
PURPLE = (128, 0, 128)
CLASS_COLOR = [BLUE, PURPLE]


def draw_detections(
        annotated_frame,
        classes,
        are_anomalous,
        boxes_corner1,
        boxes_corner2,
):
    # drawing safety circles & detection boxes
    for obj_class, is_anomaly, box_corner1, box_corner2 in zip(classes, are_anomalous, boxes_corner1, boxes_corner2):
        # Choose color depending on class (blue sheep, purple goat) and it being an anomaly (red)
        color = RED if is_anomaly else CLASS_COLOR[obj_class]
        # Draw bounding box on frame
        cv2.rectangle(annotated_frame, box_corner1, box_corner2, color, 2)


def annotate_video(
        output_dir,
        annotated_writer,
        frame,
        frame_id,
        cooldown_has_passed,
        anomaly_exists,
        classes,
        are_anomalous,
        boxes_corner1,
        boxes_corner2,
):
    annotated_frame = frame.copy()  # copy of the original frame on which to draw
    draw_detections(annotated_frame, classes, are_anomalous, boxes_corner1, boxes_corner2)

    # save the annotated frame to the video
    annotated_writer.write(annotated_frame)

    # add saving of specific frame for added context
    if cooldown_has_passed and anomaly_exists:
        # save also independent frame for improved insight
        annotated_img_path = Path(output_dir, f"anomaly_frame_{frame_id}_annotated.png")
        cv2.imwrite(annotated_img_path, annotated_frame)

import cv2
from pathlib import Path

from src.in_danger.output.frames import draw_count


RED = (0, 0, 255)
BLUE = (255, 0, 0)
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


def annotate_and_save_frame(
        output_dir,
        annotated_writer,
        frame,
        frame_id,
        raise_alert,
        num_classes,
        classes_names,
        classes,
        are_anomalous,
        boxes_corner1,
        boxes_corner2,
):
    # create copy of the original frame on which to draw
    annotated_frame = frame.copy()

    # draw detection boxes
    draw_detections(annotated_frame, classes, are_anomalous, boxes_corner1, boxes_corner2)

    # draw animal count
    draw_count(classes, num_classes, classes_names, annotated_frame)

    # save the annotated frame to the video
    annotated_writer.write(annotated_frame)

    # add saving of specific frame for added context
    if raise_alert:
        # save also independent frame for improved insight
        annotated_img_path = Path(output_dir, f"anomaly_frame_{frame_id}_annotated.png")
        cv2.imwrite(annotated_img_path, annotated_frame)

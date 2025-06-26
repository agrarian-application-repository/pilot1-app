import cv2
from pathlib import Path

from src.in_danger.output.frames import draw_count


RED = (0, 0, 255)
BLUE = (255, 0, 0)
PURPLE = (128, 0, 128)
WHITE = (255, 255, 255)
CLASS_COLOR = [BLUE, PURPLE]


def draw_detections(
        annotated_frame,
        classes,
        are_anomalous,
        ids,
        boxes_corner1,
        boxes_corner2,
):

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    text_thickness = 1

    ids_annotated_frame = annotated_frame.copy()    # create frame copy to show ids, later overlay it onto detections

    # drawing detection boxes
    for obj_class, is_anomaly, id, box_corner1, box_corner2 in zip(classes, are_anomalous, ids, boxes_corner1, boxes_corner2):
        # Choose color depending on class (purple sheep, blue goat) and it being an anomaly (red)
        color = RED if is_anomaly else CLASS_COLOR[obj_class]
        # Draw bounding box on frame
        cv2.rectangle(annotated_frame, box_corner1, box_corner2, color, 2)

        # Draw ID bounding box on frame
        cv2.rectangle(ids_annotated_frame, box_corner1, box_corner2, color, -1)        # Get text size and position
        # Setup ID text
        text = str(id)
        text_size = cv2.getTextSize(text, font, font_scale, text_thickness)[0]
        text_x = box_corner1[0] + (box_corner2[0] - box_corner1[0] - text_size[0]) // 2
        text_y = box_corner1[1] + (box_corner2[1] - box_corner1[1] + text_size[1]) // 2
        # Draw text
        cv2.putText(ids_annotated_frame, text, (text_x, text_y), font, font_scale, WHITE, text_thickness)

    # Blend overlay with original frame
    alpha = 0.4
    cv2.addWeighted(ids_annotated_frame, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)


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
        ids,
        boxes_corner1,
        boxes_corner2,
):
    # create copy of the original frame on which to draw
    annotated_frame = frame.copy()

    # draw detection boxes
    draw_detections(annotated_frame, classes, are_anomalous, ids, boxes_corner1, boxes_corner2)

    # draw animal count
    draw_count(classes, num_classes, classes_names, annotated_frame)

    # save the annotated frame to the video
    annotated_writer.write(annotated_frame)

    # add saving of specific frame for added context
    if raise_alert:
        # save also independent frame for improved insight
        annotated_img_path = Path(output_dir, f"anomaly_frame_{frame_id}_annotated.png")
        cv2.imwrite(annotated_img_path, annotated_frame)

from time import time
import numpy as np
import cv2
from pathlib import Path


RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)

CLASS_COLOR = [BLUE, PURPLE]


# generate the constant images based on the frame shape and color
def get_danger_intersect_colored_frames(shape):
    color_danger_frame = np.full(shape, RED, dtype=np.uint8)
    color_intersect_frame = np.full(shape, YELLOW, dtype=np.uint8)
    return color_danger_frame, color_intersect_frame


def draw_safety_areas(
        annotated_frame,
        boxes_centers,
        safety_radius,
):
    # drawing safety circles & detection boxes
    for box_center in boxes_centers:
        # Draw safety circle on annotated frame (green)
        cv2.circle(annotated_frame, box_center, safety_radius, GREEN, 2)


def draw_dangerous_area(
        annotated_frame,
        dangerous_mask_no_intersection,
        intersection,
        color_danger_frame,
        color_intersect_frame,
):
    # colored frames are precomputed to save time, color is always the same

    # Use cv2.bitwise_and with the masks directly.
    inter = time()
    danger_overlay = cv2.bitwise_and(color_danger_frame, color_danger_frame, mask=dangerous_mask_no_intersection)
    intersect_overlay = cv2.bitwise_and(color_intersect_frame, color_intersect_frame, mask=intersection)
    print(f"\t\tbitwise ands in {(time() - inter) * 1000:.1f} ms")

    # Combine the overlays (if regions overlap, the colors will add).
    inter = time()
    overlay = cv2.add(danger_overlay, intersect_overlay)
    print(f"\t\toverlay add in {(time() - inter) * 1000:.1f} ms")

    # Blend the overlay with the original frame.
    inter = time()
    cv2.addWeighted(annotated_frame, 0.75, overlay, 0.25, 0, annotated_frame)
    print(f"\t\taddweighted {(time() - inter) * 1000:.1f} ms")


def draw_detections(
        annotated_frame,
        classes,
        boxes_corner1,
        boxes_corner2,
):
    # drawing safety circles & detection boxes
    for obj_class, box_corner1, box_corner2 in zip(classes, boxes_corner1, boxes_corner2):
        # Draw bounding box on annotated frame (blue sheep, purple goat), on top of safety circles
        cv2.rectangle(annotated_frame, box_corner1, box_corner2, CLASS_COLOR[obj_class], 2)


def draw_count(
        classes,
        num_classes,
        classes_names,
        annotated_frame,
):
    frame_height = annotated_frame.shape[0]

    # Dynamically scale font size and thickness based on frame height
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    base_font_scale = 0.001 * frame_height  # Scale with frame height
    base_thickness = max(1, int(0.002 * frame_height))  # Ensure thickness is at least 1
    text_color = (0, 0, 0)  # BLACK
    fill_color = (255, 255, 255)  # WHITE
    line_type = cv2.LINE_AA
    org = (10, frame_height - 10)  # Initial position of the bottom-left corner of the text

    # Count classes
    class_counts = np.zeros(num_classes, dtype=np.int32)
    class_counts[: len(np.bincount(classes))] = np.bincount(classes)

    # Generate text lines
    lines = [f"N. {classes_names[idx]}: {count}" for idx, count in enumerate(class_counts)]

    # Measure text dimensions for all lines
    max_line_width = 0
    total_height = 0
    line_height = 0
    for line in lines:
        (line_width, line_height), _ = cv2.getTextSize(
            text=line,
            fontFace=font_face,
            fontScale=base_font_scale,
            thickness=base_thickness,
        )
        max_line_width = max(max_line_width, line_width)
        total_height += line_height + 5  # Add a little spacing between lines

    # Adjust text box coordinates (expand upward for all lines)
    textbox_coord_ul = (org[0] - 5, org[1] - total_height - 5)  # Expand upward
    textbox_coord_br = (org[0] + max_line_width + 5, org[1] + 5)

    # Draw white rectangle as background
    cv2.rectangle(annotated_frame, textbox_coord_ul, textbox_coord_br, fill_color, cv2.FILLED)

    # Draw each line of text inside the box
    y_offset = org[1] - total_height + line_height  # Start at the top line
    for line in lines:
        cv2.putText(
            img=annotated_frame,
            text=line,
            org=(org[0], y_offset),
            fontFace=font_face,
            fontScale=base_font_scale,
            color=text_color,
            thickness=base_thickness,
            lineType=line_type,
        )
        y_offset += line_height + 5  # Move down to the next line

    return annotated_frame


def annotate_and_save_frame(
        annotated_writer,
        output_dir,
        frame,
        frame_id,
        cooldown_has_passed,
        danger_exists,
        num_classes,
        classes_names,
        classes,
        boxes_centers,
        boxes_corner1,
        boxes_corner2,
        safety_radius_pixels,
        danger_mask,
        intersection_mask,
        color_danger_frame,
        color_intersect_frame,
):
    """ Additional annotations if videos are to be saved, or for frames where danger exist (74 ms)
    Optimization Opportunities:
    1. Batching Disk Writes:
    Disk I/O is one of the slowest parts of the process. Writing files frame by frame can be inefficient, especially if you’re saving many images.
    Solution: Buffer the frames (e.g., accumulate them in memory) and write them to disk periodically, or use a background thread/process for I/O.
    2. Avoid Repeated Path Creation:
    The Path object creation is relatively lightweight, but it can add up in tight loops.
    Solution: Pre-compute constant paths or reusable parts of the path.
    3. Optimize cv2.imwrite:
    cv2.imwrite is slower because it compresses images before saving.
    Solution: Use less compression or switch to a faster image format like .bmp if file size isn’t critical.
    4. Parallelize Save Operations:
    Writing frames and images can be offloaded to a background thread or separate process to avoid blocking the main execution.
    """

    crono_start = time()

    inter = time()
    annotated_frame = frame.copy()  # copy of the original frame on which to draw
    print(f"\tFrame copy generated in {(time() - inter) * 1000:.1f} ms")

    # draw safety circles
    inter = time()
    draw_safety_areas(annotated_frame, boxes_centers, safety_radius_pixels)
    print(f"\tsafety areas generated in {(time() - inter) * 1000:.1f} ms")

    # Overlay dangerous areas (in red) and intersections (in yellow) on the annotated frame

    inter = time()
    draw_dangerous_area(annotated_frame, danger_mask, intersection_mask, color_danger_frame, color_intersect_frame)
    print(f"\tDangerous areas AND danger INTERSECTION drawn in {(time() - inter) * 1000:.1f} ms")

    # draw detection boxes
    inter = time()
    draw_detections(annotated_frame, classes, boxes_corner1, boxes_corner2)
    print(f"\tDetections drawn in {(time() - inter) * 1000:.1f} ms")

    # save single image for better identify the exact frame if danger exists
    inter = time()
    if cooldown_has_passed and danger_exists:
        annotated_img_path = Path(output_dir, f"danger_frame_{frame_id}_annotated.jpg")
        cv2.imwrite(annotated_img_path, annotated_frame)
    print(f"\tImg saving completed in {(time() - inter) * 1000:.1f} ms")

    # draw animal count
    inter = time()
    draw_count(classes, num_classes, classes_names, annotated_frame)
    print(f"\tAnimal Count drawn in {(time() - inter) * 1000:.1f} ms")

    # save the annotated frame into video
    inter = time()
    annotated_writer.write(annotated_frame)
    print(f"\tFrame saving completed in {(time() - inter) * 1000:.1f} ms")

    print(f"Video annotations completed in {(time() - crono_start) * 1000:.1f} ms")
import multiprocessing as mp
import numpy as np
import logging
from time import time
from src.danger_detection.output.frames import (draw_count, draw_dangerous_area,
                                         draw_detections, draw_safety_areas,
                                         get_danger_intersect_colored_frames)
from src.danger_detection.processes.messages import DangerDetectionResults
from src.shared.processes.messages import AnnotationResults

# ================================================================

logger = logging.getLogger("main.danger_annotation")

if not logger.handlers:  # Avoid duplicate handlers
    video_handler = logging.FileHandler('./logs/danger_annotation.log', mode='w')
    video_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(video_handler)
    logger.setLevel(logging.DEBUG)

# ================================================================


def annotate_frame(
    frame: np.ndarray,
    num_classes: int,
    classes_names: list,
    classes: np.ndarray,
    boxes_centers: np.ndarray,
    boxes_corner1: np.ndarray,
    boxes_corner2: np.ndarray,
    safety_radius_pixels: int,
    danger_mask: np.ndarray,
    intersection_mask: np.ndarray,
    color_danger_frame: np.ndarray,
    color_intersect_frame: np.ndarray,
):

    annotated_frame = frame.copy()  # copy of the original frame on which to draw

    # draw safety circles
    if safety_radius_pixels > 0:
        draw_safety_areas(annotated_frame, boxes_centers, safety_radius_pixels)

    # Overlay dangerous areas (in red) and intersections (in yellow) on the annotated frame
    draw_dangerous_area(annotated_frame, danger_mask, intersection_mask, color_danger_frame, color_intersect_frame)

    # draw detection boxes
    draw_detections(annotated_frame, classes, boxes_corner1, boxes_corner2)

    # draw animal count
    draw_count(classes, num_classes, classes_names, annotated_frame)

    return annotated_frame


class AnnotationWorker(mp.Process):

    def __init__(self, input_queue, stream_queues, video_info_dict):
        super().__init__()

        self.video_info_dict = video_info_dict

        self.input_queue = input_queue
        self.stream_queues = stream_queues

    def run(self):

        logger.info("Annotation process started.")
        frame_width = self.video_info_dict["frame_width"]
        frame_height = self.video_info_dict["frame_height"]
        frame_shape = (frame_height, frame_width, 3)
        color_danger_frame, color_intersect_frame = get_danger_intersect_colored_frames(shape=frame_shape)
        logger.info("Running...")

        while True:
            iter_start = time()
            previous_step_results: DangerDetectionResults = self.input_queue.get()
            if previous_step_results is None:
                # Send termination signal to all stream queues
                for stream_queue in self.stream_queues:
                    stream_queue.put(None)  # Signal end of processing
                logger.info("Found sentinel value on queue. Terminating annotation process.")
                break

            annotated_frame = annotate_frame(
                    previous_step_results.frame,
                    previous_step_results.num_classes,
                    previous_step_results.classes_names,
                    previous_step_results.classes,
                    previous_step_results.boxes_centers,
                    previous_step_results.boxes_corner1,
                    previous_step_results.boxes_corner2,
                    previous_step_results.safety_radius_pixels,
                    previous_step_results.danger_mask,
                    previous_step_results.intersection_mask,
                    color_danger_frame,
                    color_intersect_frame,
            )

            result = AnnotationResults(
                frame_id=previous_step_results.frame_id,
                annotated_frame=annotated_frame,
                alert_msg=previous_step_results.danger_types,
                timestamp=previous_step_results.timestamp
            )

            # Send result to both streaming queues
            for stream_queue in self.stream_queues:
                stream_queue.put(result)

            logger.debug(f"frame {previous_step_results.frame_id}  completed in {(time()-iter_start)*1000:.2f} ms")


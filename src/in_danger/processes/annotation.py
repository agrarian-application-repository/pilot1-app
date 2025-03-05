import multiprocessing as mp
import numpy as np

from src.in_danger.output.frames import (draw_count, draw_dangerous_area,
                                         draw_detections, draw_safety_areas,
                                         get_danger_intersect_colored_frames)
from src.in_danger.processes.results import (AnnotationResults,
                                             DangerDetectionResults)


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
    draw_safety_areas(annotated_frame, boxes_centers, safety_radius_pixels)

    # Overlay dangerous areas (in red) and intersections (in yellow) on the annotated frame
    draw_dangerous_area(annotated_frame, danger_mask, intersection_mask, color_danger_frame, color_intersect_frame)

    # draw detection boxes
    draw_detections(annotated_frame, classes, boxes_corner1, boxes_corner2)

    # draw animal count
    draw_count(classes, num_classes, classes_names, annotated_frame)

    return annotated_frame


class AnnotationWorker(mp.Process):

    def __init__(self, input_queue, stream_queues, shared_dict):
        super().__init__()

        self.shared_dict = shared_dict

        self.input_queue = input_queue
        self.stream_queues = stream_queues

    def run(self):

        frame_width = self.shared_dict["frame_width"]
        frame_height = self.shared_dict["frame_height"]
        frame_shape = (frame_height, frame_width, 3)
        color_danger_frame, color_intersect_frame = get_danger_intersect_colored_frames(shape=frame_shape)

        while True:
            previous_step_results: DangerDetectionResults = self.input_queue.get()
            if previous_step_results is None:
                # Send termination signal to all stream queues
                for stream_queue in self.stream_queues:
                    stream_queue.put(None)  # Signal end of processing
                print("Terminating annotation process.")
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
                danger_types=previous_step_results.danger_types
            )

            # Send result to both streaming queues
            for stream_queue in self.stream_queues:
                stream_queue.put(result)


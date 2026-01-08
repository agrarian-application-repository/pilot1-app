import multiprocessing as mp
import numpy as np
from queue import Empty as QueueEmptyException
import logging
from time import time
from src.danger_detection.output.frames import (
    draw_count,
    draw_dangerous_area,
    draw_detections,
    draw_safety_areas,
    get_danger_intersect_colored_frames,
)
import cv2
from src.danger_detection.processes.messages import DangerDetectionResults
from src.shared.processes.messages import AnnotationResults
from src.shared.processes.constants import *

# ================================================================

logger = logging.getLogger("main.annotation")

if not logger.handlers:  # Avoid duplicate handlers
    video_handler = logging.FileHandler('./logs/annotation.log', mode='w')
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

    def __init__(
            self,
            input_queue: mp.Queue,
            video_stream_queue: mp.Queue,
            alerts_stream_queue: mp.Queue,
            error_event: mp.Event,
            alerts_cooldown_seconds: int|float,
            queue_get_timeout: float = ANNOTATION_QUEUE_GET_TIMEOUT,
            queue_put_timeout: float = ANNOTATION_QUEUE_PUT_TIMEOUT,
            max_consecutive_failures: int = ANNOTATION_MAX_CONSECUTIVE_FAILURES,
            max_put_alert_consecutive_failures: int = ANNOTATION_MAX_PUT_ALERT_CONSECUTIVE_FAILURES,
            max_put_video_consecutive_failures: int = ANNOTATION_MAX_PUT_VIDEO_CONSECUTIVE_FAILURES,
            poison_pill_timeout: float = POISON_PILL_TIMEOUT,
    ):
        super().__init__()

        self.input_queue = input_queue

        self.video_stream_queue = video_stream_queue
        self.alerts_stream_queue = alerts_stream_queue
        self.stream_queues = [
            self.video_stream_queue,
            self.alerts_stream_queue,
        ]

        self.error_event = error_event

        self.cooldown = alerts_cooldown_seconds

        self.queue_get_timeout = queue_get_timeout
        self.queue_put_timeout = queue_put_timeout / 2  # split total between 2 queues

        self.max_consecutive_failures = max_consecutive_failures
        self.max_put_alert_consecutive_failures = max_put_alert_consecutive_failures
        self.max_put_video_consecutive_failures = max_put_video_consecutive_failures

        self.poison_pill_timeout = poison_pill_timeout

    def run(self):

        logger.info("Annotation process started.")
        poison_pill_received = False

        # lazily init on first frame (need frame size)
        color_danger_frame = None       # red mask
        color_intersect_frame = None    # yellow mask

        failures = 0
        consecutive_failures = 0

        put_alert_failures = 0
        put_alert_consecutive_failures = 0

        put_video_failures = 0
        put_video_consecutive_failures = 0

        last_alert_received_timestamp = -np.inf

        logger.info("Running...")

        try:

            while not self.error_event.is_set():

                iter_start_time = time()

                # ==========================================
                # =============== INPUT FETCHING ===========
                # ==========================================

                fetch_start_time = time()

                # whether to send the alert must be evaluated each frame
                # thus, flag must be reset for every iteration = every frame (when frame in input queue)
                send_alert = False

                try:
                    # previous_step_results is either a DangerDetectionResults or the poison_pill
                    previous_step_results: DangerDetectionResults|str = self.input_queue.get(timeout=self.queue_get_timeout)
                except QueueEmptyException:
                    logger.debug("Input queue empty, retrying data fetch ... (previous process too slow or stuck?)")
                    continue    # Go back and try to read again from queue, also check the error event condition

                fetch_time = time() - fetch_start_time

                if isinstance(previous_step_results, str) and previous_step_results == POISON_PILL:
                    poison_pill_received = True
                    logger.info("Found sentinel value on queue.")
                    try:
                        logger.info("Attempting to put sentinel value on output queues ...")
                        for pidx, out_queue in enumerate(self.stream_queues, 1):
                            out_queue.put(POISON_PILL, timeout=self.poison_pill_timeout)
                            logger.info(f"Sentinel value has been passed on to downstream process #{pidx}.")
                    except Exception as e:
                        logger.error(f"Error propagating Poison Pill to one or more of the output queues: {e}")
                        self.error_event.set()
                        logger.warning(
                            "Error event set: force-stop application since downstream processes "
                            "are unable to receive the poison pill."
                        )
                    break

                # ==========================================
                # =============== DATA PROCESSING ==========
                # ==========================================

                processing_start_time = time()

                # lazy initialization of masks on first frame
                if color_danger_frame is None and color_intersect_frame is None:
                    frame_shape = previous_step_results.shape
                    color_danger_frame, color_intersect_frame = get_danger_intersect_colored_frames(shape=frame_shape)

                try:
                    annotated_frame = annotate_frame(
                            frame=previous_step_results.frame,
                            num_classes=previous_step_results.num_classes,
                            classes_names=previous_step_results.classes_names,
                            classes=previous_step_results.classes,
                            boxes_centers=previous_step_results.boxes_centers,
                            boxes_corner1=previous_step_results.boxes_corner1,
                            boxes_corner2=previous_step_results.boxes_corner2,
                            safety_radius_pixels=previous_step_results.safety_radius_pixels,
                            danger_mask=previous_step_results.danger_mask,
                            intersection_mask=previous_step_results.intersection_mask,
                            color_danger_frame=color_danger_frame,
                            color_intersect_frame=color_intersect_frame,
                    )

                    annotated_frame = cv2.resize(
                        src=annotated_frame,
                        dsize=previous_step_results.original_wh,    # (w,h)
                        interpolation=UPSAMPLING_MODE,
                    )

                    result = AnnotationResults(
                        frame_id=previous_step_results.frame_id,
                        annotated_frame=annotated_frame,
                        alert_msg=previous_step_results.danger_types,   # str
                        timestamp=previous_step_results.timestamp
                    )

                    # Check if alert should be sent
                    cooldown_has_passed = (previous_step_results.timestamp - last_alert_received_timestamp) > self.cooldown
                    alert_exist = len(previous_step_results.danger_types) > 0

                    if cooldown_has_passed and alert_exist:
                        send_alert = True
                        last_alert_received_timestamp = previous_step_results.timestamp

                    # reset processing consecutive failures counter
                    consecutive_failures = 0

                except Exception as e:
                    failures += 1
                    consecutive_failures += 1
                    logger.error(f"Failed to annotate the frame: {e}")
                    if consecutive_failures <= self.max_consecutive_failures:
                        logger.warning(
                            f"Consecutive failures: {consecutive_failures} "
                            f"(max {self.max_consecutive_failures}). "
                            f"Total failures: {failures}. "
                            f"Attempting annotation of next frame ..."
                        )
                        continue
                    else:
                        logger.critical("Max consecutive processing failures threshold passed")
                        self.error_event.set()
                        logger.warning("Error event set: force-stopping the application ")
                        break

                processing_time = time() - processing_start_time

                # ==========================================
                # =============== RESULTS PROPAGATION ======
                # ==========================================

                # if processing concludes successfully:
                # ==> pass the result to the downstream queues
                propagate_start_time = time()

                # to the alert stream queue, pass complete annotation object only if an alert should be sent (flag)
                try:
                    if send_alert:
                        self.alerts_stream_queue.put(result, timeout=self.queue_put_timeout)
                        put_alert_consecutive_failures = 0
                except Exception as e:
                    put_alert_failures += 1
                    put_alert_consecutive_failures += 1
                    logger.error(f"Failed to send alert to next process: {e}")
                    if put_alert_consecutive_failures <= self.max_put_alert_consecutive_failures:
                        logger.warning(
                            f"Consecutive failures: {put_alert_consecutive_failures} "
                            f"(max {self.max_put_alert_consecutive_failures}). "
                            f"Total failures: {put_alert_failures}. "
                            f"Attempting to send the next alert .."
                        )
                        continue
                    else:
                        logger.critical("Max consecutive alert sending failures threshold passed")
                        self.error_event.set()
                        logger.warning("Error event set: force-stopping the application")
                        break

                # to the video stream queue, pass all frames
                # prefer to push the frame without waiting, drop the frame if necessary
                try:
                    self.video_stream_queue.put_nowait(annotated_frame)
                    put_video_consecutive_failures = 0
                except Exception as e:
                    put_video_failures += 1
                    put_video_consecutive_failures += 1
                    logger.error(f"Failed to send annotated frame to next process: {e}")
                    if put_video_consecutive_failures <= self.max_put_video_consecutive_failures:
                        logger.warning(
                            f"Consecutive failures: {put_video_consecutive_failures} "
                            f"(max {self.max_put_video_consecutive_failures}). "
                            f"Total failures: {put_video_failures}. "
                            f"Attempting to send the annotated frame .."
                        )
                        continue
                    else:
                        logger.critical("Max consecutive frame sending failures threshold passed")
                        self.error_event.set()
                        logger.warning("Error event set: force-stopping the application")
                        break

                propagate_time = time() - propagate_start_time
                iter_time = time() - iter_start_time

                # monitor performance
                logger.debug(
                    f"frame {result.frame_id} processed in {iter_time * 1000:.3f} ms, "
                    f"of which --> "
                    f"FETCH: {fetch_time * 1000:.3f} ms, "
                    f"PROCESS: {processing_time * 1000:.3f} ms, "
                    f"PROPAGATE: {propagate_time * 1000:.3f} ms."
                )

        except Exception as e:
            logger.critical(f"An unexpected critical error happended in process: {e}")
            self.error_event.set()
            logger.warning("Error event set: force-stopping the application")

        finally:
            # log process conclusion
            logger.info(
                "Danger annotation process terminated successfully."
                f"Poison pill received: {poison_pill_received}. "
                f"Error event: {self.error_event.is_set()}."
            )

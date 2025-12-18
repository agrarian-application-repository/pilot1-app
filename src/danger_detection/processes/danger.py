import multiprocessing as mp
import logging
from src.danger_detection.utils import create_dangerous_intersections_masks
from src.danger_detection.processes.messages import DangerDetectionResults, DetectionResult, SegmentationResult, GeoResult
from src.shared.processes.constants import *
from time import time, sleep


# ================================================================

logger = logging.getLogger("main.danger_detection")

if not logger.handlers:  # Avoid duplicate handlers
    video_handler = logging.FileHandler('./logs/danger_detection.log', mode='w')
    video_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(video_handler)
    logger.setLevel(logging.DEBUG)

# ================================================================


class DangerDetectionWorker(mp.Process):

    def __init__(
            self,
            input_queues: list[mp.Queue],
            result_queue: mp.Queue,
            error_event: mp.Event,
            queue_get_timeout: float = MODELS_QUEUE_GET_TIMEOUT,
            queue_put_timeout: float = MODELS_QUEUE_PUT_TIMEOUT,
            poison_pill_timeout: float = POISON_PILL_TIMEOUT,
            max_consecutive_failures: int = ANNOTATION_MAX_CONSECUTIVE_FAILURES,
    ):
        super().__init__()

        self.input_queues = input_queues
        self.result_queue = result_queue
        self.error_event = error_event

        self.queue_get_timeout = queue_get_timeout
        self.queue_put_timeout = queue_put_timeout
        self.poison_pill_timeout = poison_pill_timeout

        self.max_consecutive_failures = max_consecutive_failures

    def run(self):

        logger.info("Danger detection process started.")
        poison_pill_received = False

        # lazy init on first valid frame
        frame_height = frame_width = None

        failures = 0
        consecutive_failures = 0

        while not self.error_event.is_set():
            
            iter_start = time()

            # do not collect results until all queues have at least one entry
            # short sleep in between cehcks
            any_empty = any([in_queue.Empty] for in_queue in self.input_queues)
            if any_empty:
                logger.debug(f"At least on input queue empty. Retrying in {self.queue_get_timeout} seconds")
                sleep(self.queue_get_timeout)
                continue
            
            # Collect one result from each model's result queue
            detection_result: DetectionResult = self.input_queues[0].get_nowait()
            segmentation_result: SegmentationResult = self.input_queues[1].get_nowait()
            geo_result: GeoResult = self.input_queues[2].get_nowait()

            # check whether any of the results is a poison pill
            one_is_poison_pill = (
                    detection_result == POISON_PILL or
                    segmentation_result == POISON_PILL or
                    geo_result == POISON_PILL
            )

            # if it is, propagate it and leave the loop
            if one_is_poison_pill:
                poison_pill_received = True
                logger.info("Found sentinel value on queue.")
                try:
                    logger.info("Attempting to put sentinel value on output queue ...")
                    self.result_queue.put(POISON_PILL, timeout=self.poison_pill_timeout)
                    logger.info("Sentinel value has been passed on to the next process.")
                except Exception as e:
                    logger.error(f"Error propagating Poison Pill: {e}")
                    self.error_event.set()
                    logger.warning(
                        "Error event set: "
                        "force-stop downstream processes since they are unable to receive the poison pill."
                    )
                break

            # check whether the frames are not ID aligned (critical error, set error event and terminate)
            not_frame_id_aligned = not (
                detection_result.frame_id == segmentation_result.frame_id == geo_result.frame_id
            )
            if not_frame_id_aligned:
                logger.critical(f"Model results are not frame aligned")
                self.error_event.set()
                logger.warning("Error event set: force-stopping the application")

            if frame_height is None and frame_width is None:
                frame_height, frame_width, _ = detection_result.frame.shape

            try:
                danger_mask, intersection_mask, danger_types = create_dangerous_intersections_masks(
                    frame_height=frame_height,
                    frame_width=frame_width,
                    boxes_centers=detection_result.boxes_centers,
                    safety_radius_pixels=geo_result.safety_radius_pixels,
                    segment_roads_danger_mask=segmentation_result.roads_mask,
                    segment_vehicles_danger_mask=segmentation_result.vehicles_mask,
                    dem_nodata_danger_mask=geo_result.nodata_dem_mask,
                    geofencing_danger_mask=geo_result.geofencing_mask,
                    slope_danger_mask=geo_result.slope_mask,
                )
                danger_exists = len(danger_types) > 0
                danger_type_str = " & ".join(danger_types) if danger_exists else ""

                result = DangerDetectionResults(
                    frame_id=detection_result.frame_id,
                    frame=detection_result.frame,
                    classes_names=detection_result.classes_names,
                    num_classes=detection_result.num_classes,
                    classes=detection_result.classes,
                    boxes_centers=detection_result.boxes_centers,
                    boxes_corner1=detection_result.boxes_corner1,
                    boxes_corner2=detection_result.boxes_corner2,
                    safety_radius_pixels=geo_result.safety_radius_pixels,
                    danger_mask=danger_mask,
                    intersection_mask=intersection_mask,
                    danger_types=danger_type_str,
                    timestamp=detection_result.timestamp,
                    original_wh=detection_result.original_wh,
                )

                self.result_queue.put(result, timeout=self.queue_put_timeout)
                logger.debug(f"frame {detection_result.frame_id} processed in {(time() - iter_start) * 1000:.2f} ms")
                consecutive_failures = 0

            except Exception as e:
                failures += 1
                consecutive_failures += 1
                logger.error(f"Failed to generate result or to put it in queue: {e}")
                if consecutive_failures <= self.max_consecutive_failures:
                    logger.warning(
                        f"Consecutive failures: {consecutive_failures} "
                        f"(max {self.max_consecutive_failures}). "
                        f"Total failures: {failures}. "
                        f"Attempting processing of next frame ..."
                    )
                    continue
                else:
                    logger.critical("Max consecutive processing failures threshold passed")
                    self.error_event.set()
                    logger.warning("Error event set: force-stopping the application ")
                    break   # just to be sure, would start and exit the loop immediately anyway

        # log process conclusion
        logger.info(
            "Danger detection process terminated successfully."
            f"Poison pill received: {poison_pill_received}. "
            f"Error event: {self.error_event.is_set()}."
        )

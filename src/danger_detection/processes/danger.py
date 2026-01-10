import multiprocessing as mp
import logging
from queue import Empty as QueueEmptyException
from queue import Full as QueueFullException
from src.danger_detection.utils import create_dangerous_intersections_masks
from src.danger_detection.processes.messages import DangerDetectionResults, ModelsAlignmentResult
from src.shared.processes.constants import *
from time import time


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
            input_queue: mp.Queue,
            result_queue: mp.Queue,
            error_event: mp.Event,
            queue_get_timeout: float = MODELS_QUEUE_GET_TIMEOUT,
            queue_put_timeout: float = MODELS_QUEUE_PUT_TIMEOUT,
            poison_pill_timeout: float = POISON_PILL_TIMEOUT,
    ):
        super().__init__()

        self.input_queue = input_queue
        self.result_queue = result_queue
        self.error_event = error_event

        self.queue_get_timeout = queue_get_timeout
        self.queue_put_timeout = queue_put_timeout
        self.poison_pill_timeout = poison_pill_timeout

        self.work_finished = mp.Event()

    def run(self):

        logger.info("Danger detection process started.")
        poison_pill_received = False

        # lazy init on first valid frame
        frame_height = frame_width = None

        failures = 0
        consecutive_failures = 0

        try:

            while not self.error_event.is_set():

                iter_start = time()

                try:
                    # previous_step_results is either a ModelsAlignmentResult or the poison_pill
                    previous_step_results: ModelsAlignmentResult | str = self.input_queue.get(timeout=self.queue_get_timeout)
                except QueueEmptyException:
                    logger.debug(
                        "Input queue empty, retrying data fetch ... "
                        "(Previous process too slow or stuck?)"
                    )
                    continue  # Go back and try to read again from input queue, also check the error event condition

                if isinstance(previous_step_results, str) and previous_step_results == POISON_PILL:
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
                    # exit the outer loop and terminate the process execution

                # rename for readability
                detection_result = previous_step_results.detection_result
                segmentation_result = previous_step_results.segmentation_result
                geo_result = previous_step_results.geo_result

                # lazy-init on first frame
                if frame_height is None and frame_width is None:
                    frame_height, frame_width, _ = detection_result.frame.shape

                # models outputs combining and processing
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

                try:
                    self.result_queue.put(result, timeout=self.queue_put_timeout)
                    logger.debug(
                        f"Put danger identification results for frame {detection_result.frame_id} on output queue"
                    )
                except QueueFullException:
                    logger.error(
                        f"Failed to put danger identification result for frame {detection_result.frame_id} on output queue. "
                        "Output queue is full, consumer too slow? "
                        "Discarding frame."
                    )

                logger.debug(f"frame {detection_result.frame_id} processed in {(time() - iter_start) * 1000:.2f} ms")

        except Exception as e:
            logger.critical(f"An unexpected critical error happened in danger identification process: {e}")
            self.error_event.set()
            logger.warning("Error event set: force-stopping the application")

        finally:
            # log process conclusion
            logger.info(
                "Danger identification process terminated successfully."
                f"Poison pill received: {poison_pill_received}. "
                f"Error event: {self.error_event.is_set()}."
            )
            self.work_finished.set()


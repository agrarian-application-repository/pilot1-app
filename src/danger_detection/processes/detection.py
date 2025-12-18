import multiprocessing as mp
from queue import Empty

from src.danger_detection.detection.detection import postprocess_detection_results
from ultralytics import YOLO

from src.danger_detection.processes.messages import DetectionResult
from src.shared.processes.messages import CombinedFrameTelemetryQueueObject
from src.shared.processes.constants import *
from time import time
import logging

# ================================================================

logger = logging.getLogger("main.danger_detector")

if not logger.handlers:  # Avoid duplicate handlers
    video_handler = logging.FileHandler('./logs/animals_detection.log', mode='w')
    video_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(video_handler)
    logger.setLevel(logging.DEBUG)
# ================================================================


class DetectorWrapper:

    def __init__(self, model, predict_args):
        self.detector = model
        self.predict_args = predict_args

    def predict(self, frame):
        detection_results = self.detector.predict(source=frame, verbose=False, **self.predict_args)
        return postprocess_detection_results(detection_results)


class DetectionWorker(mp.Process):
    """
    DetectionWorker is a standalone process that:
    - Instantiates the detector once during initialization
    - Stores 'detection_args' for consistent use
    - Processes incoming frames and sends back results
    - Shuts down when it receives a poison pill, forwarding the termination signal to the next process in the sequence
    """

    def __init__(
            self,
            input_queue: mp.Queue,
            result_queue: mp.Queue,
            error_event: mp.Event,
            detection_args,
            queue_get_timeout: float = MODELS_QUEUE_GET_TIMEOUT,
            queue_put_timeout: float = MODELS_QUEUE_PUT_TIMEOUT,
            poison_pill_timeout: float = POISON_PILL_TIMEOUT,
    ):
        super().__init__()

        self.input_queue = input_queue
        self.result_queue = result_queue

        self.error_event = error_event

        self.detection_args = detection_args

        self.queue_get_timeout = queue_get_timeout
        self.queue_put_timeout = queue_put_timeout
        self.poison_pill_timeout = poison_pill_timeout

    def run(self):
        """
        Main loop of the process: initializes the detector and processes frames.
        """
        
        logger.info("Animal detection process started.")
        poison_pill_received = False

        try:
            detection_model_checkpoint = self.detection_args.pop("model_checkpoint")
            model = YOLO(detection_model_checkpoint, task="detect")
            detector = DetectorWrapper(model=model, predict_args=self.detection_args)
            logger.info("Animal detection model loaded.")
            # prepare detection classes names and number
            classes_names = model.names  # Dictionary of class names
            num_classes = len(classes_names)
        except Exception as e:
            logger.critical(f"Failed to initialize the Object detection model: {e}")
            self.error_event.set()
            logger.warning("Error event set: Force-stopping all processes")

        while not self.error_event.is_set():
            
            iter_start = time()

            try:
                # frame_telemetry_object is either a CombinedFrameTelemetryQueueObject or the POISON_PILL
                frame_telemetry_object: CombinedFrameTelemetryQueueObject = self.input_queue.get(timeout=self.queue_get_timeout)
            except Empty:
                logger.debug(f"Input queue timed out. Upstream producer may be stalled. Retrying...")
                continue  # Go back and try to get again
            
            if frame_telemetry_object == POISON_PILL:
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
                        "Error event set: force-stop application since downstream processes "
                        "are unable to receive the poison pill."
                    )
                break

            get_time = time() - iter_start

            try:
                # Perform detection using stored arguments
                predict_start = time()
                classes, boxes_centers, boxes_corner1, boxes_corner2 = detector.predict(frame_telemetry_object.frame)
                predict_time = time() - predict_start
                
                result = DetectionResult(
                    frame_id=frame_telemetry_object.frame_id,
                    frame=frame_telemetry_object.frame,
                    classes_names=classes_names,
                    num_classes=num_classes,
                    classes=classes,
                    boxes_centers=boxes_centers,
                    boxes_corner1=boxes_corner1,
                    boxes_corner2=boxes_corner2,
                    timestamp=frame_telemetry_object.timestamp,
                    original_wh=frame_telemetry_object.original_wh,
                )

                # put result in output queue
                append_start = time()
                try:
                    self.result_queue.put(result, timeout=self.queue_put_timeout)
                    logger.debug("Put detection results on output queue")
                except Exception as e:
                    logger.error(
                        f"Failed to put detection results on output queue: {e}. "
                        "This might break sync between models in the next process and cause an global error event"
                    )
                append_time = time() - append_start

                iter_time = time()-iter_start

                logger.debug(
                    f"frame {frame_telemetry_object.frame_id} processed in {iter_time * 1000:.2f} ms, "
                    f"of which --> "
                    f"GET: {get_time * 1000:.2f} ms, "
                    f"PREDICT: {predict_time * 1000:.2f} ms, "
                    f"PROPAGATE: {append_time * 1000:.2f} ms."
                )
                # iteration completed correctly, move on to process next frame

            except Exception as e:
                logger.error(
                    f"Unforeseen detection error: {e}. "
                    "This might break sync between models in the next process and cause an global error event"
                )
                # continue on to next frame, next process will handle any sync error

        # log process conclusion
        logger.info(
            "Animal detection process terminated successfully."
            f"Poison pill received: {poison_pill_received}. "
            f"Error event: {self.error_event.is_set()}."
        )

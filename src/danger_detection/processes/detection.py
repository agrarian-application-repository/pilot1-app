import multiprocessing as mp

from src.danger_detection.detection.detection import postprocess_detection_results
from ultralytics import YOLO

from src.danger_detection.processes.messages import DetectionResult
from src.shared.processes.messages import CombinedFrametelemetryQueueObject
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
    - Instantiates the detector **once** during initialization
    - Stores `detection_args` for consistent use
    - Processes incoming frames and sends back results
    """

    def __init__(self, detection_args, input_queue, result_queue):
        super().__init__()
        self.detection_args = detection_args
        self.input_queue = input_queue
        self.result_queue = result_queue

    def run(self):
        """Main loop of the process: initializes the detector and processes frames."""
        logger.info("Animal detection process started.")
        detection_model_checkpoint = self.detection_args.pop("model_checkpoint")
        model = YOLO(detection_model_checkpoint, task="detect")
        logger.info("Animal detection model loaded.")
        # prepare detection classes names and number
        classes_names = model.names  # Dictionary of class names
        num_classes = len(classes_names)
        detector = DetectorWrapper(model=model, predict_args=self.detection_args)
        logger.info("Running...")

        while True:
            iter_start = time()
            frame_telemetry_object: CombinedFrametelemetryQueueObject = self.input_queue.get()
            
            if frame_telemetry_object is None:
                self.result_queue.put(None)  # Signal end of processing
                logger.info("Found sentinel value on queue. Terminating object detection process.")
                break

            # Perform detection using stored arguments
            classes, boxes_centers, boxes_corner1, boxes_corner2 = detector.predict(frame_telemetry_object.frame)
            result = DetectionResult(
                frame_id=frame_telemetry_object.frame_id,
                frame=frame_telemetry_object.frame,
                classes_names=classes_names,
                num_classes=num_classes,
                classes=classes,
                boxes_centers=boxes_centers,
                boxes_corner1=boxes_corner1,
                boxes_corner2=boxes_corner2,
                timestamp=frame_telemetry_object.timestamp
            )
            self.result_queue.put(result)

            logger.debug(f"frame {frame_telemetry_object.frame_id} completed in {(time() - iter_start) * 1000:.2f} ms")


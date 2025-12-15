import multiprocessing as mp
from queue import Empty

from src.danger_detection.detection.detection import postprocess_detection_results
from ultralytics import YOLO

from src.danger_detection.processes.messages import DetectionResult
from src.shared.processes.messages import CombinedFrameTelemetryQueueObject
from src.shared.processes.constants import POISON_PILL, QUEUE_GET_TIMEOUT
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

    def __init__(self, detection_args, input_queue, result_queue):
        super().__init__()
        self.detection_args = detection_args
        self.input_queue = input_queue
        self.result_queue = result_queue

    def run(self):
        """
        Main loop of the process: initializes the detector and processes frames.
        """
        
        logger.info("Animal detection process started.")
        detection_model_checkpoint = self.detection_args.pop("model_checkpoint")
        model = YOLO(detection_model_checkpoint, task="detect")
        detector = DetectorWrapper(model=model, predict_args=self.detection_args)
        logger.info("Animal detection model loaded.")
        
        # prepare detection classes names and number
        classes_names = model.names  # Dictionary of class names
        num_classes = len(classes_names)
        
        logger.info("Running...")

        while True:
            
            iter_start = time()

            try:
                # frame_telemetry_object is either a CombinedFrameTelemetryQueueObject or the POISON_PILL
                frame_telemetry_object: CombinedFrameTelemetryQueueObject  = self.input_queue.get(timeout=QUEUE_GET_TIMEOUT)
            except Empty:
                logger.warning(f"Input queue timed out ({QUEUE_GET_TIMEOUT} secsonds). Upstream producer may be stalled. Retrying...")
                continue # Go back and try to get again
            
            if frame_telemetry_object == POISON_PILL:
                logger.info("Found sentinel value on queue.")
                # Signal end of processing to the next process in the chain
                # Ensure the poison pill is passed on by using a blocking put (will wait until queue is free) 
                self.result_queue.put(POISON_PILL)  
                logger.info("Sentinel value has been passed on to the next process. Terminating the animal detection process...")
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
                # blocking put ensures the result is put on the target queue eventually (when space becomes available)
                # cannot discard this processing since the AI modules results must be combined later, and all resuls must be present
                append_start = time()
                self.result_queue.put(result)
                append_time = time() - append_start

                iter_time = time()-iter_start

                logger.debug(
                    f"frame {frame_telemetry_object.frame_id} processed in {iter_time * 1000:.2f} ms, "
                    f"of which --> "
                    f"GET: {get_time * 1000:.2f} ms, "
                    f"PREDICT: {predict_time * 1000:.2f} ms, "
                    f"PROPAGATE: {append_time * 1000:.2f} ms."
                )
                # iteration comleted correcly, move on to process next frame
            
            except Exception as e:
                logger.critical(f"UNFORESEEN DETECTION ERROR {e}. Shutting down ...")
                self.result_queue.put(POISON_PILL)  
                logger.info("Sentinel value has been passed on to the next process. Terminating the animal detection process...")
                break
                # TODO: how to address shutting down of prior processes?
            
        # end of process, log conclusion
        logger.info("Animal detection process terminated successfully.")
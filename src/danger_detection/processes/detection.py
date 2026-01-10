import multiprocessing as mp
from queue import Empty as QueueEmptyException
from queue import Full as QueueFullException

from src.danger_detection.detection.detection import postprocess_detection_results
from ultralytics import YOLO

from src.danger_detection.processes.messages import DetectionResult
from src.shared.processes.messages import CombinedFrameTelemetryQueueObject
from src.shared.processes.constants import *
from time import time, sleep
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

        self.work_finished = mp.Event()

    def run(self):
        """
        Main loop of the process: initializes the detector and processes frames.
        """
        
        logger.info("Animal detection process started.")
        poison_pill_received = False

        try:

            # instantiate the detection model
            detection_model_checkpoint = self.detection_args.pop("model_checkpoint")
            model = YOLO(detection_model_checkpoint, task="detect")
            detector = DetectorWrapper(model=model, predict_args=self.detection_args)
            logger.info("Animal detection model loaded.")

            # prepare detection classes names and number
            classes_names = model.names  # Dictionary of class names
            num_classes = len(classes_names)

            while not self.error_event.is_set():

                iter_start = time()

                try:
                    # frame_telemetry_object is either a CombinedFrameTelemetryQueueObject or the POISON_PILL
                    frame_telemetry_object: CombinedFrameTelemetryQueueObject | str = self.input_queue.get(timeout=self.queue_get_timeout)
                except QueueEmptyException:
                    logger.debug(f"Input queue timed out. Upstream producer may be stalled. Retrying...")
                    continue  # Go back and try to get again

                if isinstance(frame_telemetry_object, str) and frame_telemetry_object == POISON_PILL:
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
                except QueueFullException:
                    logger.error(
                        f"Failed to put detection results on output queue: queue is full. "
                        f"Consumer too slow or stuck?. "
                        f"Skipping detection results. "
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
            logger.critical(f"An unexpected critical error happened in the animal detection process: {e}")
            self.error_event.set()
            logger.warning("Error event set: force-stopping the application")

        finally:

            logger.info(
                "Animal detection process terminated successfully. "
                f"Poison pill received: {poison_pill_received}. "
                f"Error event: {self.error_event.is_set()}."
            )
            self.work_finished.set()


            """
            logger.info("Finally block")
            # Always signal that this process is done writing to the queue
            self.result_queue.close()
            logger.info("Queue closed")

            if self.error_event.is_set():
                # Case A: We know there is an error.
                # Drop everything and leave.
                # Prevents process from getting stuck while waiting for a (possibly) dead consumer to consume data
                # from the queue (might never happen if downstream process is truly dead)
                logger.info("Error event is set, cancelling output queue feeder thread data flushing...")
                self.result_queue.cancel_join_thread()

            else:
                # Case B: We think everything is fine, and the POISON PILL  was put in the output queue
                # HOWEVER: the consumer might still die before reading the poison pill, in that case this process will
                # hang forever waiting for the output queue to drain, which will never happen

                # THEREFORE:
                # Explicitly wait for the buffer to flush,
                # BUT keep an eye on the error_event while we wait.
                feeder_thread = getattr(self.result_queue, '_thread', None)
                if feeder_thread is not None and feeder_thread.is_alive():
                    logger.info("No error event receiver. Waiting for output queue feeder thread to flush...")
                    while feeder_thread.is_alive() and (not self.error_event.is_set()):
                        logger.info(f"error event: {self.error_event.is_set()}")
                        logger.info(f"feeder alive: {feeder_thread.is_alive()}")
                        sleep(0.5)

                    # If an error happened while we were waiting to flush,
                    # stop waiting and drop the buffer, just exit.
                    if self.error_event.is_set():
                        self.result_queue.cancel_join_thread()
                        logger.info(
                            "Error event was set while flushing. "
                            "Cancelling output queue feeder thread data flushing..."
                        )
                    else:
                        logger.info("Flushing of data to the output queue complete. ")

                else:
                    # If feeder_thread is None, the queue was never used.
                    # No need to wait or cancel; we can just exit.
                    logger.info(
                        "Output queue was never used or is already flushed. "
                    )
                    
            # log process conclusion
            logger.info(
                "Animal detection process terminated successfully. "
                f"Poison pill received: {poison_pill_received}. "
                f"Error event: {self.error_event.is_set()}."
            )
            """



if __name__ == "__main__":

    import numpy as np
    from src.shared.processes.consumer import Consumer
    from src.shared.processes.producer import Producer
    from src.configs.utils import read_yaml_config

    VSLOW = 1
    SLOW = 10
    FAST = 50
    REAL = 30

    CONSUMER_QUEUE_MAX = 10

    detection_args = read_yaml_config("configs/danger_detection/detector.yaml")

    def generate_frame_telemetry_queue_object():
        ts = time()
        return CombinedFrameTelemetryQueueObject(
            frame_id=int(ts*100),
            frame=np.random.randint(0, 256, size=(720, 1280, 3), dtype=np.uint8),
            telemetry=None,
            timestamp=ts,
            original_wh=(1920, 1080),
        )

    frame_telemetry_queue = mp.Queue()
    stop_event = mp.Event()
    error_event = mp.Event()

    out_queue = mp.Queue(maxsize=CONSUMER_QUEUE_MAX)

    producer = Producer(frame_telemetry_queue, error_event, generate_frame_telemetry_queue_object, frequency_hz=SLOW)
    consumer = Consumer(out_queue, error_event, frequency_hz=SLOW)

    detector = DetectionWorker(frame_telemetry_queue, out_queue, error_event, detection_args)

    print("CONSUMERS STARTED")
    consumer.start()

    sleep(3)

    print("DETECTOR STARTED")
    detector.start()

    sleep(3)

    print("PRODUCER STARTED")
    producer.start()

    sleep(5)

    #print("PRODUCER STOPPED")
    #producer.stop()
    print("ERROR EVENT SET")
    error_event.set()

    sleep(5)

    producer.join(timeout=5)
    print("producer joined")

    detector.join()
    print("detector joined")

    consumer.join()
    print("consumer joined")

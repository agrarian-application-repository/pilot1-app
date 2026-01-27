import multiprocessing as mp
from queue import Empty as QueueEmptyException
from queue import Full as QueueFullException

import logging
from src.danger_detection.segmentation.segmentation import create_onnx_segmentation_session, perform_segmentation

from src.danger_detection.processes.messages import SegmentationResult
from src.shared.processes.messages import CombinedFrameTelemetryQueueObject
from src.shared.processes.constants import *

from time import time, sleep

# ================================================================

logger = logging.getLogger("main.danger_segmentation")

if not logger.handlers:  # Avoid duplicate handlers
    video_handler = logging.FileHandler('./logs/danger_segmentation.log', mode='w')
    video_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(video_handler)
    logger.setLevel(logging.DEBUG)

# ================================================================


class SegmenterWrapper:

    def __init__(self, onnx_session, onnx_input_name, onnx_input_shape, predict_args):
        self.onnx_session = onnx_session
        self.onnx_input_name = onnx_input_name
        self.onnx_input_shape = onnx_input_shape
        self.predict_args = predict_args

    def predict(self, frame):
        segment_results = perform_segmentation(
            session=self.onnx_session,
            input_name=self.onnx_input_name,
            frame=frame,
            segmentation_args=self.predict_args
        )
        return segment_results


class SegmentationWorker(mp.Process):
    """
    SegmentationWorker is a standalone process that:
    - Instantiates the segmenter once during initialization
    - Stores 'segmentation_args' for consistent use
    - Processes incoming frames and sends back results
    - Shuts down when it receives a poison pill, forwarding the termination signal to the next process in the sequence
    """

    def __init__(
            self,
            input_queue: mp.Queue,
            result_queue: mp.Queue,
            error_event: mp.Event,
            segmentation_args,
            queue_get_timeout: float = MODELS_QUEUE_GET_TIMEOUT,
            queue_put_timeout: float = MODELS_QUEUE_PUT_TIMEOUT,
            poison_pill_timeout: float = POISON_PILL_TIMEOUT,
    ):
        super().__init__()

        self.input_queue = input_queue
        self.result_queue = result_queue

        self.error_event = error_event

        self.segmentation_args = segmentation_args

        self.queue_get_timeout = queue_get_timeout
        self.queue_put_timeout = queue_put_timeout
        self.poison_pill_timeout = poison_pill_timeout

        self.work_finished = mp.Event()

    def run(self):
        """
        Main loop of the process: initializes the detector and processes frames.
        """
        logger.info("Roads & Vehicles segmentation process started")
        poison_pill_received = False

        try:
            # initializes the segmenter
            segmentation_model_checkpoint = self.segmentation_args.pop("model_checkpoint")
            logger.info("Segmentation model checkpoint loaded")
            segmenter_session, segmenter_input_name, segmenter_input_shape = create_onnx_segmentation_session(segmentation_model_checkpoint)
            logger.info("ONNX session created")
            segmenter = SegmenterWrapper(segmenter_session, segmenter_input_name, segmenter_input_shape, self.segmentation_args)
            logger.info("Initialized segmentation model")

            while not self.error_event.is_set():

                iter_start = time()

                try:
                    # frame_telemetry_object is either a CombinedFrameTelemetryQueueObject or the POISON_PILL
                    frame_telemetry_object: CombinedFrameTelemetryQueueObject | str = self.input_queue.get(timeout=self.queue_get_timeout)
                except QueueEmptyException:
                    logger.debug(f"Input queue timed out. Upstream producer may be stalled. Retrying...")
                    continue    # Go back and try to get again

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

                # Perform segmentation using stored arguments
                predict_start = time()
                roads_mask, vehicles_mask = segmenter.predict(frame_telemetry_object.frame)
                predict_time = time() - predict_start

                # append result on next process queue
                result = SegmentationResult(
                    frame_id=frame_telemetry_object.frame_id,
                    roads_mask=roads_mask,
                    vehicles_mask=vehicles_mask,
                )

                # put result in output queue
                append_start = time()
                try:
                    self.result_queue.put(result, timeout=self.queue_put_timeout)
                    logger.debug("Put segmentation results on output queue")
                except QueueFullException:
                    logger.error(
                        f"Failed to put segmentation results on output queue: queue is full. "
                        f"Consumer too slow or stuck?. "
                        f"Skipping segmentation results. "
                        "This might break sync between models in the next process and cause an global error event"
                    )
                append_time = time() - append_start

                iter_time = time() - iter_start

                logger.debug(
                    f"frame {frame_telemetry_object.frame_id} processed in {iter_time * 1000:.2f} ms, "
                    f"of which --> "
                    f"GET: {get_time * 1000:.2f} ms, "
                    f"PREDICT: {predict_time * 1000:.2f} ms, "
                    f"PROPAGATE: {append_time * 1000:.2f} ms."
                )
                # iteration completed correctly, move on to process next frame

        except Exception as e:
            logger.critical(f"An unexpected critical error happened in the segmentation process: {e}",  exc_info=True)
            self.error_event.set()
            logger.warning("Error event set: force-stopping the application")

        finally:
            # log process conclusion
            logger.info(
                "Roads & vehicles segmentation process terminated successfully. "
                f"Poison pill received: {poison_pill_received}. "
                f"Error event: {self.error_event.is_set()}."
            )
            self.work_finished.set()


if __name__ == "__main__":

    import numpy as np
    from src.shared.processes.consumer import Consumer
    from src.shared.processes.producer import Producer
    from src.configs.utils import read_yaml_config

    VSLOW = 1
    SLOW = 10
    FAST = 50
    REAL = 30

    CONSUMER_QUEUE_MAX = 2

    segmentation_args = read_yaml_config("configs/danger_detection/segmenter.yaml")

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

    producer = Producer(frame_telemetry_queue, error_event, generate_frame_telemetry_queue_object, frequency_hz=FAST)
    consumer = Consumer(out_queue, error_event, frequency_hz=VSLOW)

    segmenter = SegmentationWorker(frame_telemetry_queue, out_queue, error_event, segmentation_args)

    print("CONSUMERS STARTED")
    consumer.start()

    sleep(3)

    print("DETECTOR STARTED")
    segmenter.start()

    sleep(3)

    print("PRODUCER STARTED")
    producer.start()

    sleep(3)

    print("PRODUCER STOPPED")
    producer.stop()

    sleep(3)

    # print("POISON PILL")
    # frame_telemetry_queue.put(POISON_PILL)    # option1
    print("ERROR EVENT")
    error_event.set()  # option2

    producer.join()
    print("producer joined")
    segmenter.join()
    print("detector joined")

    print("CONSUMER STOPPED")
    consumer.join()
    consumer.join()
    print("consumer joined")


import multiprocessing as mp
from queue import Empty

import logging
from src.danger_detection.segmentation.segmentation import create_onnx_segmentation_session, perform_segmentation

from src.danger_detection.processes.messages import SegmentationResult
from src.shared.processes.messages import CombinedFrameTelemetryQueueObject
from src.shared.processes.constants import POISON_PILL, QUEUE_GET_TIMEOUT

from time import time

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
            input_shape=self.onnx_input_shape,
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

    def __init__(self, segmentation_args, input_queue, result_queue):
        super().__init__()
        self.segmentation_args = segmentation_args
        self.input_queue = input_queue
        self.result_queue = result_queue

    def run(self):
        logger.info("Segmentation process started")
        # initializes the segmenter
        segmentation_model_checkpoint = self.segmentation_args.pop("model_checkpoint")
        logger.info("Segmentation model checkpoint loaded")
        segmenter_session, segmenter_input_name, segmenter_input_shape = create_onnx_segmentation_session(segmentation_model_checkpoint)
        logger.info("ONNX session created")
        segmenter = SegmenterWrapper(segmenter_session, segmenter_input_name, segmenter_input_shape, self.segmentation_args)
        logger.info("Initialized segmentation model")
        
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
                logger.info("Sentinel value has been passed on to the next process. Terminating roads & vehicles segentation process...")
                break

            get_time = time() - iter_start

            try:
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
                # blocking put ensures the result is put on the target queue eventually (when space becomes available)
                # cannot discard this processing since the AI modules results must be combined later, and all resuls must be present
                append_start = time()
                self.result_queue.put(result)
                append_time = time() - append_start

                iter_time = time() - iter_start

                logger.debug(
                    f"frame {frame_telemetry_object.frame_id} processed in {iter_time * 1000:.2f} ms, "
                    f"of which --> "
                    f"GET: {get_time * 1000:.2f} ms, "
                    f"PREDICT: {predict_time * 1000:.2f} ms, "
                    f"PROPAGATE: {append_time * 1000:.2f} ms."
                )
                # iteration comleted correcly, move on to process next frame

            except Exception as e:
                logger.critical(f"UNFORESEEN SEGMENTATION ERROR {e}. Shutting down ...")
                self.result_queue.put(POISON_PILL)  
                logger.info("Sentinel value has been passed on to the next process. Terminating the animal detection process...")
                break
                # TODO: how to address shutting down of prior processes?

        # end of process, log conclusion
        logger.info("Roads & vehicles segmentation process terminated successfully.")

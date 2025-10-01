import multiprocessing as mp
import logging
from src.danger_detection.segmentation.segmentation import create_onnx_segmentation_session, perform_segmentation

from src.danger_detection.processes.messages import SegmentationResult
from src.shared.processes.messages import CombinedFrametelemetryQueueObject
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
    - Instantiates the segmenter **once** during initialization
    - Stores `segmentation_args` for consistent use
    - Processes incoming frames and sends back results
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

        # start processing frames
        while True:
            iter_start = time()
            frame_telemetry_object: CombinedFrametelemetryQueueObject = self.input_queue.get()
            if frame_telemetry_object is None:
                self.result_queue.put(None)  # Signal end of processing
                logger.info("Found sentinel value on queue. Terminating segmentation process.")
                break

            # Perform segmentation using stored arguments
            predict_start = time()
            roads_mask, vehicles_mask = segmenter.predict(frame_telemetry_object.frame)
            predict_time = time() - predict_start

            # append result on next process queue
            append_start = time()
            result = SegmentationResult(
                frame_id=frame_telemetry_object.frame_id,
                roads_mask=roads_mask,
                vehicles_mask=vehicles_mask,
            )
            self.result_queue.put(result)
            append_time = time() - append_start

            logger.debug("Appended mask to next process queue")

            logger.debug(f"frame {frame_telemetry_object.frame_id} predict completed in {predict_time*1000:.2f} ms")
            logger.debug(f"frame {frame_telemetry_object.frame_id} append completed in {append_time*1000:.2f} ms")
            logger.debug(f"frame {frame_telemetry_object.frame_id} completed in {(time()-iter_start)*1000:.2f} ms")


import multiprocessing as mp

from src.in_danger.segmentation.segmentation import \
    postprocess_segmentation_results
from ultralytics import YOLO

from src.in_danger.processes.results import FrameQueueObject, SegmentationResult


class SegmenterWrapper:

    def __init__(self, model, predict_args):
        self.segmenter = model
        self.predict_args = predict_args

    def predict(self, frame):
        segment_results = self.segmenter.predict(source=frame, verbose=False, **self.predict_args)
        # frame size (H, W, 3)
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        return postprocess_segmentation_results(segment_results, frame_height, frame_width)


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
        """Main loop of the process: initializes the segmenter and processes frames."""
        segmentation_model_checkpoint = self.segmentation_args.pop("model_checkpoint")
        model = YOLO(segmentation_model_checkpoint, task="segment")
        segmenter = SegmenterWrapper(model=model, predict_args=self.segmentation_args)

        while True:
            frame_object: FrameQueueObject = self.input_queue.get()
            if frame_object is None:
                self.result_queue.put(None)  # Signal end of processing
                print("Terminating segmentation process.")
                break

            # Perform segmentation using stored arguments
            mask = segmenter.predict(frame_object.frame)
            result = SegmentationResult(frame_id=frame_object.frame_id, mask=mask)
            self.result_queue.put(result)

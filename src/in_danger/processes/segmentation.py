import multiprocessing as mp

from src.in_danger.segmentation.segmentation import create_onnx_segmentation_session, perform_segmentation

from src.in_danger.processes.results import FrameQueueObject, SegmentationResult


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
        """Main loop of the process: initializes the segmenter and processes frames."""
        segmentation_model_checkpoint = self.segmentation_args.pop("model_checkpoint")
        segmenter_session, segmenter_input_name, segmenter_input_shape = create_onnx_segmentation_session(segmentation_model_checkpoint)
        segmenter = SegmenterWrapper(segmenter_session, segmenter_input_name, segmenter_input_shape, self.segmentation_args)

        while True:
            frame_object: FrameQueueObject = self.input_queue.get()
            if frame_object is None:
                self.result_queue.put(None)  # Signal end of processing
                print("Terminating segmentation process.")
                break

            # Perform segmentation using stored arguments
            result = segmenter.predict(frame_object.frame)
            self.result_queue.put(result)

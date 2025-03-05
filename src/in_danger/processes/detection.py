import multiprocessing as mp

from src.in_danger.detection.detection import postprocess_detection_results
from ultralytics import YOLO

from src.in_danger.processes.results import DetectionResult, FrameQueueObject


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
        detection_model_checkpoint = self.detection_args.pop("model_checkpoint")
        model = YOLO(detection_model_checkpoint, task="detect")
        # prepare detection classes names and number
        classes_names = model.names  # Dictionary of class names
        num_classes = len(classes_names)
        detector = DetectorWrapper(model=model, predict_args=self.detection_args)

        while True:
            frame_object: FrameQueueObject = self.input_queue.get()
            if frame_object is None:
                self.result_queue.put(None)  # Signal end of processing
                print("Terminating object detection process.")
                break

            # Perform detection using stored arguments
            classes, boxes_centers, boxes_corner1, boxes_corner2 = detector.predict(frame_object.frame)
            result = DetectionResult(
                frame_id=frame_object.frame_id,
                frame=frame_object.frame,
                classes_names=classes_names,
                num_classes=num_classes,
                classes=classes,
                boxes_centers=boxes_centers,
                boxes_corner1=boxes_corner1,
                boxes_corner2=boxes_corner2,
            )
            self.result_queue.put(result)

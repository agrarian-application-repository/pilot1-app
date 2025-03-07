import multiprocessing as mp

import cv2
import time

from src.in_danger.processes.results import FrameQueueObject


class VideoReader(mp.Process):
    """Reads video frames and pushes them to the frame queue."""
    def __init__(self, source, models_queues, shared_dict, video_info_set_event):
        super().__init__()

        self.source = source
        self.models_input_queues = models_queues

        self.shared_dict = shared_dict
        self.video_info_set_event = video_info_set_event

    def run(self):
        cap = cv2.VideoCapture(self.source)  # Open webcam or video file
        if not cap.isOpened():
            print("Error: Unable to open video source. Terminating video reading process.")
            # Signal that video information is (attempted to be) set
            self.video_info_set_event.set()
            # Send termination signal to all model queues
            for model_queue in self.models_input_queues:
                model_queue.put(None)  # Signal end of processing
            return

        # set application-wide info about the video stream
        self.shared_dict["frame_width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.shared_dict["frame_height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.shared_dict["fps"] = int(cap.get(cv2.CAP_PROP_FPS))
        # ensure other processes don't start until these are set
        self.video_info_set_event.set()

        # Initialize a counter for frame IDs
        frame_id = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:     # End of video or read error
                # Send termination signal to all model queues
                for model_queue in self.models_input_queues:
                    model_queue.put(None)  # Signal end of processing
                cap.release()   # release video reader
                print("Terminating video reading process.")
                break  # process terminates

            # Package the frame with its unique frame ID
            frame_object = FrameQueueObject(frame_id=frame_id, frame=frame)
            # Distribute the same frame to each detector's input queue
            for model_queue in self.models_input_queues:
                model_queue.put(frame_object)
            frame_id += 1

            time.sleep(0.025)   # 25ms delay to simulate real time stream



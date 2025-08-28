import multiprocessing as mp
import logging
import cv2
import time
from pathlib import Path
from src.shared.processes.messages import CombinedFrametelemetryQueueObject
from src.shared.drone_utils.flight_logs import parse_drone_flight_data

# ================================================================

logger = logging.getLogger("main.files_reader")

if not logger.handlers:  # Avoid duplicate handlers
    video_handler = logging.FileHandler('./logs/files_reader.log', mode='w')
    video_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(video_handler)
    logger.setLevel(logging.DEBUG)

# ================================================================


class VideoTelemetryFilesReader(mp.Process):
    """Reads video frames and corresponding telemetry, and pushes them to the next processes queue queue."""
    def __init__(
            self,
            source:str,
            telemetry_file:str,
            models_queues,
            video_info_dict,
            video_info_set_event
    ):
        super().__init__()

        self.video_source = source
        self.telemetry_file = telemetry_file
        self.models_input_queues = models_queues

        self.video_info_dict = video_info_dict
        self.video_info_set_event = video_info_set_event

    def run(self):
        
        # Open drone flight data
        flight_data_file_path = Path(self.telemetry_file)
        flight_data_file = open(flight_data_file_path, "r")
        
        if not flight_data_file:
            logger.error("Unable to open video source. Terminating video reading process.")
            # Signal that video information is (attempted to be) set
            self.video_info_set_event.set()
            # Send termination signal to all model queues
            for model_queue in self.models_input_queues:
                model_queue.put(None)  # Signal end of processing
            return

        cap = cv2.VideoCapture(self.video_source)  # Open webcam or video file
        if not cap.isOpened():
            logger.error("Unable to open video source. Terminating video reading process.")
            # Signal that video information is (attempted to be) set
            self.video_info_set_event.set()
            # Send termination signal to all model queues
            for model_queue in self.models_input_queues:
                model_queue.put(None)  # Signal end of processing
            return

        # set application-wide info about the video stream
        self.video_info_dict["frame_width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_info_dict["frame_height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_info_dict["fps"] = int(cap.get(cv2.CAP_PROP_FPS))
        # ensure other processes don't start until these are set
        self.video_info_set_event.set()

        # Initialize a counter for frame IDs
        frame_id = 1
        
        # ietarte over frames
        while cap.isOpened():
            success, frame = cap.read()
            
            if not success:     # End of video or read error
                # Send termination signal to all model queues
                for model_queue in self.models_input_queues:
                    model_queue.put(None)  # Signal end of processing
                cap.release()   # release video reader
                print("Terminating video and telemetry reading process.")
                break  # process terminates

            frame_flight_data = parse_drone_flight_data(flight_data_file, frame_id)

            # Package the frame with its unique frame ID
            frame_telemetry_object = CombinedFrametelemetryQueueObject(
                frame_id=frame_id, 
                frame=frame,
                telemetry=frame_flight_data,
                timestamp=time.time(),
            )
            # Distribute the same frame to each detector's input queue
            for model_queue in self.models_input_queues:
                model_queue.put(frame_telemetry_object)
            
            frame_id += 1

            time.sleep(0.025)   
            # 25ms delay to simulate (fast) real time stream and assess wheter folowing processes can keep up

        flight_data_file.close()



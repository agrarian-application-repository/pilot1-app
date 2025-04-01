import multiprocessing as mp
from pathlib import Path
import time

import cv2
import msgpack

from src.in_danger.processes.results import AnnotationResults


class VideoStreamWriter(mp.Process):
    def __init__(self, output_dir, input_queue, shared_dict):
        super().__init__()
        self.shared_dict = shared_dict
        self.input_queue = input_queue
        self.output_dir = Path(output_dir)
        self.frame_times = []  # Store timestamps for FPS calculation

    def run(self):

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logfile = open(self.output_dir / "video_stream_writer_log.txt", "w")

        annotated_video_path = self.output_dir / "annotated_stream.mp4"
        out = cv2.VideoWriter(
            filename=annotated_video_path,
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=self.shared_dict["fps"],
            frameSize=(self.shared_dict["frame_width"], self.shared_dict["frame_height"])
        )

        while True:
            previous_step_results: AnnotationResults = self.input_queue.get()

            if previous_step_results is None:
                out.release()
                logfile.write("Terminating output video streaming process.")
                logfile.close()
                break  # End of processing

            # save frame to video stream
            out.write(previous_step_results.annotated_frame)  # Save frame to video

            # Track FPS
            self.frame_times.append(time.time())
            if len(self.frame_times) > 30:  # Use last 30 frames to calculate FPS
                self.frame_times.pop(0)

            if len(self.frame_times) > 1:
                avg_fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])
                logfile.write(f"Stream VIDEO frame {previous_step_results.frame_id + 1} | FPS: {avg_fps:.2f}")
            else:
                logfile.write(f"Stream VIDEO frame {previous_step_results.frame_id + 1}")



class NotificationsStreamWriter(mp.Process):
    def __init__(self, output_dir, cooldown_seconds, input_queue, shared_dict):
        super().__init__()
        self.shared_dict = shared_dict
        self.input_queue = input_queue
        self.output_dir = Path(output_dir)
        self.last_alert_frame_id = -shared_dict["fps"]
        self.alerts_frames_cooldown = shared_dict["fps"] * cooldown_seconds

    def run(self):

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logfile = open(self.output_dir / "notification_stream_writer_log.txt", "w")

        while True:
            previous_step_results: AnnotationResults = self.input_queue.get()

            if previous_step_results is None:
                logfile.write("Terminating output notification streaming process.")
                logfile.close()
                break  # End of processing

            # save msg-img pairs when cooldown has passed and danger is present
            cooldown_has_passed = (previous_step_results.frame_id - self.last_alert_frame_id) >= self.alerts_frames_cooldown
            danger_exists = len(previous_step_results.danger_types) > 0

            if cooldown_has_passed and danger_exists:
                self.last_alert_frame_id = previous_step_results.frame_id   # update last alert id to restart cooldown
                msgpack_data = msgpack.packb({
                    "frame_id": previous_step_results.frame_id+1,
                    "danger_types": previous_step_results.danger_types,
                    "img_bytes": previous_step_results.annotated_frame.tobytes(),
                    "img_shape": previous_step_results.annotated_frame.shape,
                    "img_dtype": str(previous_step_results.annotated_frame.dtype),
                }, use_bin_type=True)
                msgpack_path = self.output_dir / f"frame_{previous_step_results.frame_id+1}.msgpack"
                with open(msgpack_path, "wb") as f:
                    f.write(msgpack_data)

            logfile.write(f"Stream NOTIFICATION frame {previous_step_results.frame_id + 1}")

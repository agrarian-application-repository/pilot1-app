import multiprocessing as mp
from pathlib import Path
import time

import cv2
import logging
from src.shared.processes.messages import AnnotationResults

logger = logging.getLogger("main.video_writer")

# ================================================================

if not logger.handlers:  # Avoid duplicate handlers
    video_handler = logging.FileHandler('./logs/video_writer.log', mode='w')
    video_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(video_handler)
    logger.setLevel(logging.DEBUG)

# ================================================================


class VideoStreamFileWriter(mp.Process):
    def __init__(self, output_dir, input_queue, video_info_dict):
        super().__init__()
        self.video_info_dict = video_info_dict
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
            fps=self.video_info_dict["fps"],
            frameSize=(self.video_info_dict["frame_width"], self.video_info_dict["frame_height"])
        )

        while True:
            previous_step_results: AnnotationResults = self.input_queue.get()

            if previous_step_results is None:
                out.release()
                logfile.write("Terminating output video streaming process.")
                logger.info("Terminating output video streaming process.")
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
                logger.info(f"Stream VIDEO frame {previous_step_results.frame_id + 1} | FPS: {avg_fps:.2f}")
            else:
                logfile.write(f"Stream VIDEO frame {previous_step_results.frame_id + 1}")
                logger.info(f"Stream VIDEO frame {previous_step_results.frame_id + 1}")

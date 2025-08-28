import multiprocessing as mp
from pathlib import Path
from time import time
import msgpack
import logging
from src.shared.processes.messages import AnnotationResults

logger = logging.getLogger("main.msg_writer")

# ================================================================

if not logger.handlers:  # Avoid duplicate handlers
    video_handler = logging.FileHandler('./logs/msg_writer.log', mode='w')
    video_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(video_handler)
    logger.setLevel(logging.DEBUG)

# ================================================================


class NotificationsStreamFileWriter(mp.Process):
    def __init__(self, output_dir, cooldown_seconds, input_queue, video_info_dict):
        super().__init__()
        self.video_info_dict = video_info_dict
        self.input_queue = input_queue
        self.output_dir = Path(output_dir)
        self.last_alert_frame_id = -video_info_dict["fps"]
        self.alerts_frames_cooldown = video_info_dict["fps"] * cooldown_seconds

    def run(self):

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logfile = open(self.output_dir / "notification_stream_writer_log.txt", "w")

        while True:
            iter_start = time()

            previous_step_results: AnnotationResults = self.input_queue.get()

            if previous_step_results is None:
                logfile.write("Terminating output notification streaming process.")
                logfile.close()
                break  # End of processing

            # save msg-img pairs when cooldown has passed and danger is present
            cooldown_has_passed = (previous_step_results.frame_id - self.last_alert_frame_id) >= self.alerts_frames_cooldown
            alert_exists = len(previous_step_results.alert_msg) > 0

            if cooldown_has_passed and alert_exists:
                self.last_alert_frame_id = previous_step_results.frame_id   # update last alert id to restart cooldown
                msgpack_data = msgpack.packb({
                    "frame_id": previous_step_results.frame_id+1,
                    "alert_msg": previous_step_results.alert_msg,
                    "img_bytes": previous_step_results.annotated_frame.tobytes(),
                    "img_shape": previous_step_results.annotated_frame.shape,
                    "img_dtype": str(previous_step_results.annotated_frame.dtype),
                }, use_bin_type=True)
                msgpack_path = self.output_dir / f"frame_{previous_step_results.frame_id+1}.msgpack"
                with open(msgpack_path, "wb") as f:
                    f.write(msgpack_data)

            logfile.write(f"Stream NOTIFICATION frame {previous_step_results.frame_id + 1}")
            logger.info(
                f"Stream NOTIFICATION frame {previous_step_results.frame_id + 1}, "
                f"in {(time()-iter_start)*1000:.2f}")

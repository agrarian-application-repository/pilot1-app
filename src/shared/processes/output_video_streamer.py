import os
import cv2
import logging
import threading
import multiprocessing as mp
from typing import Optional
from queue import Empty as QueueEmptyException
import numpy as np

from src.shared.processes.video_stream_manager import VideoStreamManager

from src.shared.processes.constants import *


# ================================================================

logger = logging.getLogger("main.video_out")

if not logger.handlers:  # Avoid duplicate handlers
    video_handler = logging.FileHandler('./logs/video_out.log', mode='w')
    video_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(video_handler)
    logger.setLevel(logging.DEBUG)

# ================================================================


class VideoProducerProcess(mp.Process):

    def __init__(
            self,
            input_queue: mp.Queue,
            output_queue: mp.Queue,
            error_event: mp.Event,
            fps: int = FPS,
            get_frame_timeout: float = VIDEO_WRITER_GET_FRAME_TIMEOUT,
            poison_pill_timeout: float = POISON_PILL_TIMEOUT,
            # -------- LOCAL SAVE -------------
            local_video_name: str = ANNOTATED_VIDEO_NAME,
            video_codec: str = CODEC,
            # -------- STREAM MANAGER -------------
            media_server_url: Optional[str] = None,
            stream_manager_queue_max_size: int = MAX_SIZE_VIDEO_STREAM,
            stream_manager_queue_get_timeout: float = VIDEO_OUT_STREAM_QUEUE_GET_TIMEOUT,
            stream_manager_ffmpeg_startup_timeout: float = VIDEO_OUT_STREAM_FFMPEG_STARTUP_TIMEOUT,
            stream_manager_ffmpeg_shutdown_timeout: float = VIDEO_OUT_STREAM_FFMPEG_SHUTDOWN_TIMEOUT,
            stream_manager_startup_timeout: float = VIDEO_OUT_STREAM_STARTUP_TIMEOUT,
            stream_manager_shutdown_timeout: float = VIDEO_OUT_STREAM_SHUTDOWN_TIMEOUT,
            # -------- STORAGE_MANAGER ------------
            storage_manager_handoff_timeout: float = VIDEO_WRITER_HANDOFF_TIMEOUT,
    ):
        """
         A process that pulls message objects containing video frames from a 
         multiprocessing Queue where a previous process in the chain dumped its result.
         When a frame arrives, this process:
         - adds the frame to a local video file (.mp4 or avi)
         - puts the frame into the queue of a VideoStreamManager,
             which uses a thread and ffmpeg to push the frame as a video stream
             to a mediaserver (url provided) using RTMP
         The process terminates either:
         - when it receives a poison pill from the queue (msg == POISON_PILL)
         - when the global error_event is set by another process
         In both cases, a clean shutdown is expected. At shutdown:
         - the cv2.Videocapute object is closed, saving the video file locally
         - an persitent storage uploading process in the chain is informed to either start 
          uploading the video (url), or to stop because saving failed (poison_pill)
        """

        super().__init__()

        self.input_queue = input_queue
        self.output_queue = output_queue
        self.error_event = error_event

        self.fps = fps
        self.get_frame_timeout = get_frame_timeout

        # local file save config
        self.writer: Optional[cv2.VideoWriter] = None
        self.local_video_filename = local_video_name
        self.video_codec = video_codec

        # stream manager config
        self.stream_manager: Optional[VideoStreamManager] = None
        self.media_server_url = media_server_url
        self.stream_manager_queue_max_size = stream_manager_queue_max_size
        self.stream_manager_queue_get_timeout = stream_manager_queue_get_timeout
        self.stream_manager_ffmpeg_startup_timeout = stream_manager_ffmpeg_startup_timeout
        self.stream_manager_ffmpeg_shutdown_timeout = stream_manager_ffmpeg_shutdown_timeout
        self.stream_manager_startup_timeout = stream_manager_startup_timeout
        self.stream_manager_shutdown_timeout = stream_manager_shutdown_timeout

        # storage manager config
        self.storage_manager_handoff_timeout = storage_manager_handoff_timeout

        self.poison_pill_timeout = poison_pill_timeout

        self.work_finished = mp.Event()

    def _init_writer(self, width: int, height: int):
        fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
        self.writer = cv2.VideoWriter(
            filename=self.local_video_filename,
            fourcc=fourcc,
            fps=self.fps,
            frameSize=(width, height),
        )
        logger.info(f"VideoWriter instantiated. Set frame size to {width}x{height}")

    def _process_frame(self, frame: np.ndarray):

        height, width, _ = frame.shape

        # Lazy-init the VideoWriter on the first frame
        if self.writer is None:
            self._init_writer(width=width, height=height)

        # Lazy-setup&start the StreamManager thread on the first frame
        if self.stream_manager and self.stream_manager.width is None and self.stream_manager.height is None:
            self.stream_manager.set_frame_dims(width=width, height=height)
            # try to start video streaming thread. Errors are handles internally
            # if start fails, the stream manager is abandoned and the video will only be saved locally
            if not self.stream_manager.start():
                self.stream_manager = None

        # Add frame to local video file
        self.writer.write(frame)

        # Push frame to Real-Time Stream Manager
        if self.stream_manager:
            self.stream_manager.push_to_queue(frame)

    def run(self):

        logger.info("Worker process started.")

        # create the media server manager
        # setup and start is performed lazily on first frame received (W,H info needed)
        if self.media_server_url:
            self.stream_manager = VideoStreamManager(
                mediaserver_url=self.media_server_url,
                fps=self.fps,
                queue_max_size=self.stream_manager_queue_max_size,
                queue_get_timeout=self.stream_manager_queue_get_timeout,
                ffmpeg_startup_timeout=self.stream_manager_ffmpeg_startup_timeout,
                ffmpeg_shutdown_timeout=self.stream_manager_ffmpeg_shutdown_timeout,
                startup_timeout=self.stream_manager_startup_timeout,
                shutdown_timeout=self.stream_manager_shutdown_timeout,
            )
        
        try:

            while not self.error_event.is_set():

                try:
                    # Timeout allows checking the error_event periodically
                    msg = self.input_queue.get(timeout=self.get_frame_timeout)
                except QueueEmptyException:
                    logger.info("Input queue empty. Continuing to wait for frames ... ")
                    continue

                if isinstance(msg, str) and msg == POISON_PILL:
                    logger.info("Poison pill received. Shutting down...")
                    break

                self._process_frame(msg)

        except Exception as e:
            logger.error(f"Critical error in worker: {e}", exc_info=True)
            self.error_event.set()
            logger.error("Error event set: force-stopping the application")

        finally:
            self._shutdown_procedure()

    def _shutdown_procedure(self):
        """
        Ensures resources are released and triggers persistence.
        """

        save_failed = False

        if self.writer:
            try:
                self.writer.release()
                logger.info(f"The recording has been saved locally at {self.local_video_filename}")
            except Exception as e:
                save_failed = True
                logger.error(
                    f"Error saving the recording: {e}. "
                    f"Check status at {self.local_video_filename}"
                )

        # stop the video streaming thread
        # handles errors internally
        if self.stream_manager:
            self.stream_manager.stop()

        # output_queue is the input queue for the next process in the chain, the PersistenceProcess.
        # It uploads the video to an object storage in the cloud.
        # Since the application is instantiated once per requesting user, 
        # and since this is the only process writing on said queue,
        # we are sure that the queue will be empty and whatever
        # we write on it will succeed without queue-related errors
        # (queue size >= 2: path and pill -->  upload and stop )
        try:

            if not save_failed:
                self.output_queue.put(self.local_video_filename, timeout=self.storage_manager_handoff_timeout)
                logger.info("Video save success: passed path to video persistence process for upload")
            else:
                logger.info("Video save failed: path to video not passed to persistence process")

            # in any case, the Persistence Process must stop, so we pass the POISON_PILL
            self.output_queue.put(POISON_PILL, timeout=self.poison_pill_timeout)
            logger.info("Passed poison pill on to video persistence process")

        except Exception as e:
            logger.error(f"Unable to write on the PersistenceProcess input queue: {e}")

        # log definitive process termination
        logger.info(
            "All tasks assigned to VideoProducerProcess are terminated."
            "Clean shutdown: complete"
        )
        self.work_finished.set()


import subprocess
import threading
import logging
import time
from queue import Queue, Empty, Full
import numpy as np

from src.shared.processes.constants import (
    VIDEO_WRITER_FPS,
    VIDEO_OUT_STREAM_QUEUE_MAX_SIZE,
    VIDEO_OUT_STREAM_QUEUE_GET_TIMEOUT,
    VIDEO_OUT_STREAM_FFMPEG_STARTUP_TIMEOUT,    # 0.5
    VIDEO_OUT_STREAM_FFMPEG_SHUTDOWN_TIMEOUT,    # 8.0
    VIDEO_OUT_STREAM_STARTUP_TIMEOUT,           # 2.0
    VIDEO_OUT_STREAM_SHUTDOWN_TIMEOUT,           # 10.0
)

# ================================================================
logger = logging.getLogger("main.video_out.stream")

if not logger.handlers:  # Avoid duplicate handlers
    video_handler = logging.FileHandler('/app/logs/video_out_stream.log', mode='w')
    video_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(video_handler)
    logger.setLevel(logging.DEBUG)
# ================================================================


class VideoStreamManager:
    def __init__(
            self,
            mediaserver_url: str,
            fps: int = VIDEO_WRITER_FPS,
            queue_max_size: int = VIDEO_OUT_STREAM_QUEUE_MAX_SIZE,
            queue_get_timeout: float = VIDEO_OUT_STREAM_QUEUE_GET_TIMEOUT,
            ffmpeg_startup_timeout: float = VIDEO_OUT_STREAM_FFMPEG_STARTUP_TIMEOUT,
            ffmpeg_shutdown_timeout: float = VIDEO_OUT_STREAM_FFMPEG_SHUTDOWN_TIMEOUT,
            startup_timeout: float = VIDEO_OUT_STREAM_STARTUP_TIMEOUT,
            shutdown_timeout: float = VIDEO_OUT_STREAM_SHUTDOWN_TIMEOUT,
    ):

        self.mediaserver_url = mediaserver_url
        self.fps = fps

        # lazy-init frame dimensions based on first frame received
        self.width = None
        self.height = None

        self.frame_queue = Queue(maxsize=queue_max_size)

        self.running = False
        self.stream_thread = None
        self._ffmpeg_process = None

        self.queue_get_timeout = queue_get_timeout
        self.ffmpeg_startup_timeout = ffmpeg_startup_timeout
        self.ffmpeg_shutdown_timeout = ffmpeg_shutdown_timeout
        self.startup_timeout = startup_timeout
        self.shutdown_timeout = shutdown_timeout

        # Synchronization for startup health check
        self._start_confirmed = threading.Event()
        self._startup_error = None

    def set_frame_dims(self, width: int, height: int):
        self.width = width
        self.height = height

    def push_to_queue(self, frame: np.ndarray) -> bool:
        """
        Receives frames from the Worker process.
        Returns whether the frame was enqueued or not.
        """
        try:
            self.frame_queue.put_nowait(frame)
            return True
        except Full:
            logger.warning("Stream queue full, dropping frame to maintain real-time sync.")
            return False

    def _stream_loop(self):
        """
        Background thread that manages the FFmpeg pipe and ensures
        graceful termination.
        """
        # Optimized command for Full HD 30fps ingest
        command = [
            'ffmpeg',
            '-y', '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f"{self.width}x{self.height}",
            '-r', str(self.fps),
            '-i', '-',  # Input from stdin pipe
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'veryfast',
            '-tune', 'zerolatency',
            '-f', 'flv',
            self.mediaserver_url
        ]

        try:
            # bufsize is important for high bitrate 1080p to prevent pipe clogging
            self._ffmpeg_process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10 ** 7
            )

            # --- STARTUP VERIFICATION ---
            # Brief sleep to allow FFmpeg to initialize/fail connection
            time.sleep(self.ffmpeg_startup_timeout)
            if self._ffmpeg_process.poll() is not None:
                # Process exited immediately
                _, stderr_data = self._ffmpeg_process.communicate()
                self._startup_error = stderr_data.decode().split('\n')[-2] if stderr_data else "Unknown error"
                self._start_confirmed.set()
                return

            # If we reach here, process is alive
            self._start_confirmed.set()

            # Loop continues as long as we are "running"
            # OR there are frames left to drain.
            while self.running or not self.frame_queue.empty():
                try:
                    # Use a timeout to block for a short time only
                    frame = self.frame_queue.get(timeout=self.queue_get_timeout)
                    self._ffmpeg_process.stdin.write(frame.tobytes())
                except Empty:
                    logger.debug("Queue empty. Continuing ...")
                    continue
                except BrokenPipeError:
                    logger.error("FFmpeg pipe broken. Media server likely disconnected.")
                    break

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            self._startup_error = str(e)
            self._start_confirmed.set()
        finally:
            self._finalize_ffmpeg()

    def _finalize_ffmpeg(self):
        """Internal routine to close the pipe and wait for process exit."""
        if self._ffmpeg_process:
            logger.info("Draining complete. Closing FFmpeg pipe...")
            try:
                if self._ffmpeg_process.stdin:
                    self._ffmpeg_process.stdin.close()

                # Wait for FFmpeg to wrap up the FLV container
                self._ffmpeg_process.wait(timeout=self.ffmpeg_shutdown_timeout)
                logger.info("FFmpeg process exited cleanly.")
            except subprocess.TimeoutExpired:
                logger.warning("FFmpeg did not exit in time. Forcing termination.")
                self._ffmpeg_process.kill()
            except Exception as e:
                logger.error(f"Error during FFmpeg shutdown: {e}")

    def start(self) -> bool:
        """
        Launches the streaming thread and prepares the FFmpeg pipe.
        Returns True if the stream started correctly, False otherwise.
        """
        if self.running:
            logger.warning("Stream Manager is already running.")
            return True

        if self.width is None or self.height is None:
            logger.error("Cannot start StreamManager: Frame dimensions not set.")
            return False

        self._start_confirmed.clear()
        self._startup_error = None
        self.running = True

        self.stream_thread = threading.Thread(
            target=self._stream_loop,
            name="StreamThread",
            daemon=True
        )
        self.stream_thread.start()

        # Wait for the thread to confirm success (blocking start)
        # Timeout slightly longer than the sleep in the thread
        started = self._start_confirmed.wait(timeout=self.startup_timeout)

        if not started or self._startup_error:
            error_msg = self._startup_error if self._startup_error else "Timeout during startup"
            logger.error(f"Streaming failed to start: {error_msg}")
            self.stop()
            return False

        logger.info(f"Streaming thread successfully initialized for target: {self.mediaserver_url}")
        return True

    def stop(self):
        """Triggers graceful shutdown of the streaming thread."""

        # set stopping flag
        self.running = False

        # Give it a moment (timeout) to flush the queue before the parent process exits
        if self.stream_thread:
            self.stream_thread.join(timeout=self.shutdown_timeout)
            if self.stream_thread.is_alive():
                logger.warning("Video Streaming thread did not terminate cleanly within timeout")
            else:
                logger.info("Video Streaming thread terminated successfully")

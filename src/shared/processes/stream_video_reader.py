import multiprocessing as mp
import cv2
import logging
import time
from src.shared.processes.messages import FrameQueueObject

# ================================================================

logger = logging.getLogger("main.stream_video_in")

if not logger.handlers:  # Avoid duplicate handlers
    video_handler = logging.FileHandler('/app/logs/stream_video_in.log', mode='w')
    video_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(video_handler)
    logger.setLevel(logging.DEBUG)

# ================================================================

QUEUE_PUT_TIMEOUT = 0.02     # block for up to 20 ms to put data in output queue


VIDEO_STREAM_READER_EXPECTED_ASPECT_RATIO = 16.0/9.0
VIDEO_STREAM_READER_PROCESSING_SHAPE = (1280, 720)

VIDEO_STREAM_URL = "rtmp://<server>[:port]/<app>/<stream_key>"
# VIDEO_STREAM_URL = "rtmps://<server>[:port]/<app>/<stream_key>"
# VIDEO_STREAM_URL = "rtsp://[user[:password]@]host[:port]/path"
# VIDEO_STREAM_URL = "rtsps://[user[:password]@]host[:port]/path"


VIDEO_STREAM_READER_CONNECTION_OPEN_TIMEOUT_S = 5.0
VIDEO_STREAM_READER_FRAME_READ_TIMEOUT_S = 0.1
VIDEO_STREAM_READER_MAX_CONSECUTIVE_CONNECTION_FAILURES = 5


VIDEO_STREAM_READER_BUFFER_SIZE = 1

VIDEO_STREAM_READER_RECONNECT_DELAY = 5.0
VIDEO_STREAM_READER_RETRY_DELAY = 0.1

INTERPOLATION_MODE = cv2.INTER_AREA


class StreamVideoReader(mp.Process):
    """
    Reads video stream frames from media server or CDN,
    and pushes them to the frame queue together with id and timestamp.
    The process continually tries to connect to the provided server address.
    The process terminates when it receives an external stop event
    """
    
    def __init__(
            self,
            frame_queue: mp.Queue,
            stop_event: mp.Event,
        ):
        
        super().__init__()

        # Shared output queue. Next process will read from this
        self.frame_queue = frame_queue

        # Shared stop event. Allows to stop all processes at the same time
        self.stop_event = stop_event

    @staticmethod
    def _setup_capture() -> cv2.VideoCapture:
        """
        Set up video capture with appropriate settings for video streams.
        """
        cap = cv2.VideoCapture(VIDEO_STREAM_URL)
        # timeout for opening source
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, VIDEO_STREAM_READER_CONNECTION_OPEN_TIMEOUT_S * 1000)
        # timeout for reading frame
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, VIDEO_STREAM_READER_FRAME_READ_TIMEOUT_S * 1000)
        # Reduce buffer for lower latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, VIDEO_STREAM_READER_BUFFER_SIZE)
        
        return cap

    def run(self):
        """Main process loop."""

        # connection failure counters
        total_connection_failures = 0
        consecutive_connection_failures = 0

        # Initialize frame counter
        frame_id = 0

        # read failure counters
        total_read_failures = 0
        consecutive_read_failures = 0

        # placeholder for videoCapture connection
        cap = None

        # Use a non-blocking check to see if we should stop.
        while not self.stop_event.is_set():

            # ----- initialize connection --------

            # attempt to connect to the video source. 
            # A new connection is created at every failure to ensure clean state
            cap = self._setup_capture()

            # Initial connection isn't established, retry after a delay
            if not cap.isOpened():
                total_connection_failures += 1
                consecutive_connection_failures += 1
                logger.warning(
                    f"Unable to open video source {VIDEO_STREAM_URL}"
                    f"N. Consecutive connection failures: {consecutive_connection_failures} "
                    f"(max: {VIDEO_STREAM_READER_MAX_CONSECUTIVE_CONNECTION_FAILURES}). "
                    f"N. Total connection failures: {total_connection_failures}. "
                )

                # when the max number of connection attempts has not been surpassed, retry to connect
                if consecutive_connection_failures <= VIDEO_STREAM_READER_MAX_CONSECUTIVE_CONNECTION_FAILURES:
                    logger.warning(f"Retrying to connect in {VIDEO_STREAM_READER_RECONNECT_DELAY} seconds ...")
                    time.sleep(VIDEO_STREAM_READER_RECONNECT_DELAY)
                    continue
                # otherwise, stop the application
                else:
                    logger.warning(
                        f"Max number of connection attempts to the video stream source surpassed. "
                        f"Shutting down the application ..."
                    )
                    self.stop_event.set()
                    break
                    # after break, jump out of this outer loop to the cleanup code

            # ----- connection established ------------

            # reset counter of consecutive failed connection attempts
            consecutive_connection_failures = 0

            logger.info("Starting video reading loop")
            while cap.isOpened() and not self.stop_event.is_set():
                # continue to read frame until the connection is live and the stop_event is not set
                # if loop exists due to connection breakdown, the outer loop retries to establish the connection
                # if loop exits due to stop_event, outer loop exists as well, and the final cleanup code is executed

                success, frame = cap.read()
                frame_id += 1

                # --------- read failure ---------
                if (not success) or (frame is None) or (frame.size == 0):
                    total_read_failures += 1
                    consecutive_read_failures += 1
                    logger.warning(
                        f"Frame {frame_id} read failed. "
                        f"N. Consecutive read failures: {consecutive_read_failures}. "
                        f"N. Total read failures: {total_read_failures}. "
                        f"Attempting new read in {VIDEO_STREAM_READER_RETRY_DELAY} seconds... "
                    )
                    time.sleep(VIDEO_STREAM_READER_RETRY_DELAY)
                    continue

                # --------- read successful ---------

                # Reset failure counter on successful read
                consecutive_read_failures = 0

                # check that the video frames are in the expected 16/9 aspect ratio
                frame_height, frame_width, _ = frame.shape
                aspect_ratio = frame_width/frame_height
                if not abs(aspect_ratio - VIDEO_STREAM_READER_EXPECTED_ASPECT_RATIO) < 1e-6:     # tolerance
                    logger.error(
                        f"Application expects frame with aspect ratio (W/H)={VIDEO_STREAM_READER_EXPECTED_ASPECT_RATIO} "
                        f"but got frame of size W/H = {frame_width}{frame_width} = {aspect_ratio}."
                        f"Shutting down the application ..."
                    )
                    self.stop_event.set()
                    break
                    # after break, skip to the end of this inner loop,
                    # enter the outer loop with terminates due to stop_event being set.
                    # This causes a jump to the final cleanup code

                # resize to desired frame size, here (1280, 720) as a compromise between resolution and speed
                frame = cv2.resize(frame, VIDEO_STREAM_READER_PROCESSING_SHAPE, interpolation=INTERPOLATION_MODE)

                # Package the frame with its unique frame ID
                frame_object = FrameQueueObject(
                    frame_id=frame_id,
                    frame=frame,
                    timestamp=time.time()
                )

                # Try to put output object in output queue
                try:
                    self.frame_queue.put(frame_object, timeout=QUEUE_PUT_TIMEOUT)
                    logger.debug(f"Added frame {frame_id} to queue.")
                # Catch exception for queue full specifically
                except mp.queues.Full:
                    logger.warning(
                        f"Failed to put frame in queue: Queue is full. "
                        f"Consumer too slow or stopped?. "
                        f"Continuing to listen for new frames in {VIDEO_STREAM_READER_RETRY_DELAY} seconds"
                    )
                    time.sleep(VIDEO_STREAM_READER_RETRY_DELAY)
                    continue
                # handles all other exceptions
                except Exception as e:
                    logger.warning(f"Exception: {e}")
                    continue

                # ----- end of successful read of frame -----
                # move on to read next frame

        # Final Cleanup
        if cap is not None:
            cap.release()

        logger.info("StreamVideoReader process stopping")


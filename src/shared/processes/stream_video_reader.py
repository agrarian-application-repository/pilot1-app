import multiprocessing as mp
import cv2
import logging
import time
from messages import FrameQueueObject

# ================================================================

logger = logging.getLogger("main.stream_video_in")

if not logger.handlers:  # Avoid duplicate handlers
    video_handler = logging.FileHandler('/app/logs/stream_video_in.log')
    video_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(video_handler)
    logger.setLevel(logging.DEBUG)

# ================================================================


class StreamVideoReader(mp.Process):
    """Reads video frames and pushes them to the frame queue."""
    
    def __init__(
            self, 
            video_info_dict, 
            source, 
            frame_queue, 
            buffer_size=1, 
            opening_timeout_s = 10.0,
            reading_timeout_s = 5.0,
            reconnect_attempts=5,
            reconnect_delay_s=2.0,
            max_consecutive_read_fail=5
        ):
        
        super().__init__()
        self.source = source
        self.frame_queue = frame_queue
        self.video_info_dict = video_info_dict
        
        self.buffer_size = buffer_size
        self.opening_timeout_s = opening_timeout_s
        self.reading_timeout_s = reading_timeout_s
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay_s = reconnect_delay_s
        self.max_consecutive_read_fail = max_consecutive_read_fail
        
        self._stop_event = mp.Event()

    # TODO improve
    def _is_source_stream(self) -> bool:
        return isinstance(self.source, str) and ('rtsp://' in self.source or 'rtmp://' in self.source)
        
    def _setup_capture(self) -> cv2.VideoCapture:
        """Set up video capture with appropriate settings."""
        cap = cv2.VideoCapture(self.source)
        # Settings for RTSP/RTMP streams
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)  # Reduce buffer for lower latency
        # Add timeout for network streams
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.opening_timeout_s * 1000)  # timeout for opening source
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.reading_timeout_s * 1000)  # timeout for reading frame

        # Set video info
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_info_dict["frame_width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_info_dict["frame_height"])
        cap.set(cv2.CAP_PROP_FPS, self.video_info_dict["fps"])
        
        return cap

    def _attempt_reconnect(self) -> cv2.VideoCapture | None:
        """Attempt to reconnect to the video source."""
        for attempt in range(1, self.reconnect_attempts+1):
            
            # stop immediatly if a stop event is set
            if self._stop_event.is_set():
                return None
                
            logger.info(f"Reconnection attempt {attempt}/{self.reconnect_attempts} in {self.reconnect_delay_s} seconds ...")
            time.sleep(self.reconnect_delay_s)
            
            cap = self._setup_capture()
            if cap.isOpened():
                logger.info("Successfully reconnected to video source")
                return cap
            else:
                logger.error(f"Failed to reconnected on attempt #{attempt}")
                cap.release()
                
        logger.error("All reconnection attempts failed")
        return None
        
    def run(self):
        """Main process loop."""
        
        # terminate immediately if the source is invalid for a RTSP/RTMP stream
        if not self._is_source_stream():
            self._terminate_process()
            return

        # first attempt at opening the video source
        cap = self._setup_capture()
        
        # Initial connection check
        if not cap.isOpened():
            logger.error("Unable to open video source on first try")
            # Try reconnecting for streams
            cap = self._attempt_reconnect()
            if cap is None:
                self._terminate_process()
                return
        
        # Initialize frame counter
        frame_id = 0
        consecutive_failures = 0
        
        logger.info("Starting video reading loop")
        
        while cap.isOpened() and not self._stop_event.is_set():
            success, frame = cap.read()
            frame_id += 1
            
            if not success:
                
                consecutive_failures += 1
                logger.warning(f"Frame read failed (consecutive failures: {consecutive_failures})")
                
                if consecutive_failures < self.max_consecutive_read_fail:
                    # Brief pause before retry
                    time.sleep(0.1)  
                    continue

                # when the number of fails surpasses the threshold ...
                # try to reconnect (end processing if reconnection fails)
                logger.info("Too many consecutive failures, attempting reconnection")
                cap.release()
                cap = self._attempt_reconnect() 
                if cap is None:
                    break   # terminate process
                else:
                    consecutive_failures = 0
                    continue    # try to read new frame
                    
            # Reset failure counter on successful read
            consecutive_failures = 0
            
            # Validate frame
            if frame is None or frame.size == 0:
                logger.warning("Received empty frame, skipping")
                continue

            # Package the frame with its unique frame ID
            frame_object = FrameQueueObject(
                frame_id=frame_id, 
                frame=frame, 
                timestamp=time.time()
            )

            # Put frame in queue (with timeout to prevent blocking)
            try:
                self.frame_queue.put(frame_object, timeout=1.0)
                logger.debug(f"Added frame {frame_id} to queue")
            # break out of video processing if a problem arises in putting frames on queue
            except mp.queues.Full as e: # Catch Full specifically
                logger.error(f"Failed to put frame in queue: Queue is full. Consumer too slow or stopped?: {e}")
                break
            except Exception as e:
                logger.error(f"Exception: {e}")
                break

        # Cleanup
        if cap is not None:
            cap.release()
        
        self._terminate_process()
        logger.info("Video reading process terminated")
        
    def _terminate_process(self):
        """Send termination signal and set event."""
        try:
            # add sentinel value to queue
            self.frame_queue.put(None, timeout=1.0)
            # set stop event if not already set from outside
            if not self._stop_event.is_set():
                self._stop_event.set()

        except Exception as e:
            logger.error(f"Failed to send termination signal: {e}")

    def is_running(self) -> bool:
        """
        Check if the process is currently running.
        
        Returns:
            bool: True if process is alive and not stopped
        """
        return self.is_alive() and not self._stop_event.is_set()
            
    def stop(self):
        """Gracefully stop the video reading process."""
        logger.info("Stop signal received")
        self._stop_event.set()

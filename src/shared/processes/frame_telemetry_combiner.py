import multiprocessing as mp
import logging
import time
from collections import deque
from src.shared.processes.messages import FrameQueueObject, TelemetryQueueObject, CombinedFrameTelemetryQueueObject
from typing import Optional

# ================================================================

logger = logging.getLogger("main.combiner")

if not logger.handlers:  # Avoid duplicate handlers
    video_handler = logging.FileHandler('/app/logs/combiner.log', mode='w')
    video_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(video_handler)
    logger.setLevel(logging.DEBUG)

# ================================================================

QUEUE_GET_TIMEOUT = 0.01
# Timeout of 0.01s = 100Hz = fast enough to ensure that, even with waiting,
# we are able to process a 30 fps video
QUEUE_PUT_MAX_RETRIES = 3
QUEUE_PUT_RETRY_DELAY = 0.005
# 3 attemps over a max of 15 ms


class FrameTelemetryCombiner(mp.Process):
    """
    A multiprocessing Process that combines frames with telemetry data based on timestamps.

    One frame at a time (from oldest, FIFO):
    - Takes frame and its id from frame_queue (multiprocessing Queue)
    - searches in the telemetry_queue (multiprocessing Queue) the telemetry values that best match based
    on the timestamp (max_time_diff_s sets the max time difference allowed).
    Telemetry values are delivered in order of timestamp
    - if no match is found, the matching telemetry value must be set to None in the output object
    - if match is found,removes all the older telemetry values from the queue to free space
    - outputs combined data to output_queues combining frame and telemetry (ensuring the data is put on all queues,
    either on all or none, list of multiporcessing Queue objects)

    The process must be stopped based on a stop event coming from the outised to ensure graceful exiting.
    The process uses a global logger to log information, warnings and errors
    """

    def __init__(
            self,
            frame_queue: mp.Queue,
            telemetry_queue: mp.Queue,
            output_queues: list[mp.Queue],
            stop_event: mp.Event,
            max_time_diff_s: float = 0.15
    ):
        """
        Initialize the FrameTelemetryCombiner process.

        Args:
            frame_queue: Queue containing FrameQueueObject instances
            telemetry_queue: Queue containing TelemetryQueueObject instances
            output_queues: List of queues to output CombinedFrameTelemetryQueueObject instances
            stop_event: Event to signal the process to stop
            max_time_diff_s: Maximum time difference allowed for matching (seconds)
        """
        super().__init__()
        self.frame_queue = frame_queue
        self.telemetry_queue = telemetry_queue
        self.output_queues = output_queues
        self.stop_event = stop_event
        self.max_time_diff_s = max_time_diff_s
        self.telemetry_buffer = deque(maxlen=2 * telemetry_queue._maxsize)

    def run(self):
        """Main process loop."""
        logger.info("FrameTelemetryCombiner process started")

        failed_matches = 0
        consecutive_failed_matches = 0

        # Process runs until the stop event is set
        while not self.stop_event.is_set():

            try:
                # Try to get a frame, waiting for a short time if not available immediately
                frame_obj: FrameQueueObject = self.frame_queue.get(timeout=QUEUE_GET_TIMEOUT)
            except:
                # Queue remains empty or timeout expires
                # continue to check stop_event, if not set, try again to get a frame
                continue

            # Collect available telemetry data into buffer
            self._update_telemetry_buffer()

            # Find best matching telemetry
            matched_telemetry = self._find_best_match(frame_obj.timestamp, failed_matches)

            if matched_telemetry is None:
                failed_matches += 1
                consecutive_failed_matches += 1
                logger.debug(
                    f"N. Consecutive failed matches: {consecutive_failed_matches}. "
                    f"N. Total failed matches: {failed_matches}. "
                )
            else:
                consecutive_failed_matches = 0

            # Create combined object
            combined_obj = CombinedFrameTelemetryQueueObject(
                frame_id=frame_obj.frame_id,
                frame=frame_obj.frame,
                telemetry=matched_telemetry,
                timestamp=frame_obj.timestamp
            )

            # Output to all queues (all or none)
            if not self._output_to_all_queues(combined_obj):
                logger.error(f"Failed to output combined data for frame {frame_obj.frame_id}")

        logger.info("FrameTelemetryCombiner process stopping")

    def _update_telemetry_buffer(self):
        """Collect all available telemetry data from queue into buffer."""
        while True:
            try:
                telemetry_obj: TelemetryQueueObject = self.telemetry_queue.get_nowait()
                self.telemetry_buffer.append(telemetry_obj)
            except:
                # Queue is empty or telemetry_buffer is full
                break

    def _find_best_match(self, frame_timestamp: float) -> Optional[dict]:
        """
        Find the best matching telemetry for the given frame timestamp.
        Removes all older telemetry values from buffer if match is found.
        Find the best matching telemetry for the given frame timestamp.

        Args:
            frame_timestamp: Timestamp of the frame to match

        Returns:
            Matched telemetry dict or None if no match within time threshold
        """
        if not self.telemetry_buffer:
            logger.warning(f"No telemetry data available for matching at timestamp {frame_timestamp}")
            return None

        best_match = None
        best_diff = float('inf')
        best_idx = -1  # Track index of the telemetry with timestamp closest to that of the frame
        last_too_old_idx = -1  # Track oldest telemetry that is too old and should be removed

        min_valid_timestamp = frame_timestamp - self.max_time_diff_s

        # Find closest telemetry by timestamp
        for idx, telemetry_obj in enumerate(self.telemetry_buffer):
            time_diff = abs(telemetry_obj.timestamp - frame_timestamp)

            if time_diff < best_diff:
                best_diff = time_diff
                best_match = telemetry_obj
                best_idx = idx

            # Track the last telemetry that's too old (before min_valid_timestamp)
            if telemetry_obj.timestamp < min_valid_timestamp:
                last_too_old_idx = idx

            # Since telemetry is ordered by timestamp, if we're past the frame
            # timestamp by too much, we can stop searching
            if telemetry_obj.timestamp > frame_timestamp + self.max_time_diff_s:
                break

        # Remove old telemetry regardless of match success
        if last_too_old_idx >= 0:
            # Remove all telemetry up to and including last_too_old_idx
            for _ in range(last_too_old_idx + 1):
                self.telemetry_buffer.popleft()
            logger.debug(f"Removed {last_too_old_idx + 1} old telemetry entries")

            # Adjust best_idx if we removed items before it
            if best_idx >= 0:
                best_idx = best_idx - (last_too_old_idx + 1)

        # Check if best match is within allowed time difference
        if best_diff <= self.max_time_diff_s:
            logger.debug(f"Found telemetry match with time diff: {best_diff:.4f}s")
            # Remove all telemetry older than to the matched one (keep matched one)
            for _ in range(best_idx):
                self.telemetry_buffer.popleft()
            return best_match.telemetry

        else:
            logger.warning(
                f"No telemetry match found within {self.max_time_diff_s} seconds "
                f"(best diff: {best_diff:.4f}s). "
            )
            return None

    def _output_to_all_queues(self, combined_obj: CombinedFrameTelemetryQueueObject) -> bool:
        """
        Output combined data to all output queues (all or none).

        Args:
            combined_obj: The combined object to output

        Returns:
            True if successfully output to all queues, False otherwise
        """

        # Check if all queues have space (not full) before attempting to put
        # This provides atomicity guarantee
        for attempt in range(QUEUE_PUT_MAX_RETRIES):

            all_have_space = all([not queue.full() for queue in self.output_queues])

            if all_have_space:
                # All queues have space, proceed with putting
                for queue in self.output_queues:
                    queue.put_nowait(combined_obj)

                logger.debug(
                    f"Successfully output frame {combined_obj.frame_id} "
                    f"to all {len(self.output_queues)} processing queues")
                return True

            else:
                # At least one queue is full, wait and retry
                if attempt < QUEUE_PUT_MAX_RETRIES - 1:
                    logger.debug(f"Some queues full, retrying ({attempt + 1}/{QUEUE_PUT_MAX_RETRIES})...")
                    time.sleep(QUEUE_PUT_RETRY_DELAY)

        logger.warning(
            f"Failed to output frame {combined_obj.frame_id}: "
            f"queues full after {QUEUE_PUT_MAX_RETRIES} attempts")
        return False


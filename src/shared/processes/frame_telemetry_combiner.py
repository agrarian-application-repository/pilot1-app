import multiprocessing as mp
import logging
import time
from queue import Empty as QueueEmptyException
from queue import Full as QueueFullException
from collections import deque
from src.shared.processes.messages import FrameQueueObject, TelemetryQueueObject, CombinedFrameTelemetryQueueObject
from typing import Optional

from src.shared.processes.constants import *

# ================================================================

logger = logging.getLogger("main.combiner")

if not logger.handlers:  # Avoid duplicate handlers
    video_handler = logging.FileHandler('/app/logs/combiner.log', mode='w')
    video_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(video_handler)
    logger.setLevel(logging.DEBUG)

# ================================================================


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
    either on all or none, list of multiprocessing Queue objects)

    The process can shut-down via a global ErrorEvent being set (hard shutdown),
    or via POISON-PILL (sequential shutdown).
    If the process stops due to error_event, it's not necessary to propagate the poison pill since all processes will
    stop at the same time.

    """

    def __init__(
            self,
            frame_queue: mp.Queue,
            telemetry_queue: mp.Queue,
            output_queues: list[mp.Queue],
            error_event: mp.Event,
            telemetry_buffer_max_size: int = FRAMETELCOMB_MAX_TELEM_BUFFER_SIZE,
            max_time_diff_s: float = FRAMETELCOMB_MAX_TIME_DIFF,
            queue_get_timeout: float = FRAMETELCOMB_QUEUE_GET_TIMEOUT,
            queue_put_max_retries: int = FRAMETELCOMB_QUEUE_PUT_MAX_RETRIES,
            queue_put_backoff: float = FRAMETELCOMB_QUEUE_PUT_BACKOFF,
            poison_pill_backoff: float = POISON_PILL_TIMEOUT,
    ):
        """
        Initialize the FrameTelemetryCombiner process.

        Args:
            frame_queue: Queue containing FrameQueueObject instances
            telemetry_queue: Queue containing TelemetryQueueObject instances
            output_queues: List of queues to output CombinedFrameTelemetryQueueObject instances
            error_event: Event to signal the process to stop
            max_time_diff_s: Maximum time difference allowed for matching (seconds)
        """
        super().__init__()

        # mp queues and events
        self.frame_queue = frame_queue
        self.telemetry_queue = telemetry_queue
        self.output_queues = output_queues
        self.error_event = error_event

        # local telemetry buffer for timestamp matching
        self.telemetry_buffer = deque(maxlen=telemetry_buffer_max_size)
        self.max_time_diff_s = max_time_diff_s

        # queue get
        self.queue_get_timeout = queue_get_timeout

        # multi-queue put
        self.queue_put_max_retries = queue_put_max_retries
        self.queue_put_backoff = queue_put_backoff
        self.poison_pill_backoff = poison_pill_backoff

    def _update_telemetry_buffer(self):
        """Collect all available telemetry data from queue into buffer."""
        while True:
            try:
                telemetry_obj: TelemetryQueueObject = self.telemetry_queue.get_nowait()
                self.telemetry_buffer.append(telemetry_obj)
            except QueueEmptyException:
                logger.debug("Telemetry queue empty, stopping fetch.")
                break
            except QueueFullException:
                logger.debug("Telemetry buffer full, stopping fetch. Check frame-telemetry combining performance.")
                break
            except Exception as e:
                logger.debug(f"Unexpected error in telemetry fetch: {e}. Stopping fetch")
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

    def _output_to_all_queues(
            self,
            combined_obj: CombinedFrameTelemetryQueueObject|str,
            backoff: float,         # different for poison pill and frame
    ) -> bool:
        """
        Output combined data to all output queues (all or none).

        Args:
            combined_obj: The combined object to output

        Returns:
            True if successfully output to all queues, False otherwise
        """

        if combined_obj == POISON_PILL:
            obj_type = "poison pill"
        else:
            obj_type = f"frame {combined_obj.frame_id}"

        # Check if all queues have space (not full) before attempting to put
        # This provides atomicity guarantee
        for attempt in range(self.queue_put_max_retries, 1):

            all_have_space = all([not queue.full() for queue in self.output_queues])

            if all_have_space:
                # All queues have space, proceed with putting
                for queue in self.output_queues:
                    queue.put_nowait(combined_obj)

                logger.debug(
                    f"Successfully output {obj_type} "
                    f"to all {len(self.output_queues)} processing queues")
                return True

            else:
                # At least one queue is full, wait and retry
                if attempt <= self.queue_put_max_retries:
                    logger.debug(f"Some queues full, retrying ({attempt}/{self.queue_put_max_retries})...")
                    time.sleep(backoff)

        logger.warning(
            f"Failed to output put {obj_type} to model queues: "
            f"queues full after {self.queue_put_max_retries} attempts. "
            f"Discarding {obj_type}."
        )

        # if it's the poison pill we were unable to propagate, set the error event to force-stop the application
        if combined_obj == POISON_PILL:
            self.error_event.set()
            logger.error(
                "Error event set: could not propagate poison pill to all models queues."
                "Force-stopping the application"

            )
        return False

    def run(self):
        """Main process loop."""
        logger.info("FrameTelemetryCombiner process started")

        failed_matches = 0
        consecutive_failed_matches = 0
        poison_pill_received = False

        # Process runs until the stop event is set
        while not self.error_event.is_set():

            try:
                # Try to get a frame, waiting for a short time if not available immediately
                frame_obj: FrameQueueObject = self.frame_queue.get(timeout=QUEUE_GET_TIMEOUT)
            except QueueEmptyException:
                logger.debug("Frame queue is empty, retrying fetch ...")
                continue

            # if the object found is the poison pill, it must be propagated to following processes via
            # their input queues. Must ensure that all downstream processes receive the pill
            if frame_obj == POISON_PILL:
                logger.info("Found sentinel value on queue.")
                poison_pill_received = True
                # internally handles setting of error_event if unable to propagate the poison pill to all queues
                # error_event causes clean shutdown of all processes
                self._output_to_all_queues(frame_obj, backoff=self.poison_pill_backoff)
                break

            # Collect available telemetry data into buffer
            # stop when all have been collected and the queue is empty, or the local telemetry data buffer is full
            self._update_telemetry_buffer()

            # Find best matching telemetry
            matched_telemetry = self._find_best_match(frame_obj.timestamp)

            if matched_telemetry is None:
                failed_matches += 1
                consecutive_failed_matches += 1
                logger.debug(
                    f"N. Consecutive failed matches: {consecutive_failed_matches}. "
                    f"N. Total failed matches: {failed_matches}. "
                    f"Either the delay between frames and telemetry is too large, or telemetry collection has stopped."
                )
            else:
                consecutive_failed_matches = 0

            # Create combined object
            combined_obj = CombinedFrameTelemetryQueueObject(
                frame_id=frame_obj.frame_id,
                frame=frame_obj.frame,
                telemetry=matched_telemetry,
                timestamp=frame_obj.timestamp,
                original_wh=frame_obj.original_wh,
            )

            # Output to all queues (all or none = discard frame)
            self._output_to_all_queues(combined_obj, backoff=self.queue_put_backoff)

        # log process conclusion
        logger.info(
            "FrameTelemetryCombiner process stopped gracefully."
            f"Poison pill received: {poison_pill_received}. "
            f"Error event: {self.error_event.is_set()}."
        )


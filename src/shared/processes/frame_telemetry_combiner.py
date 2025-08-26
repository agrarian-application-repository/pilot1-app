import multiprocessing as mp
import logging
from collections import deque
from src.shared.processes.messages import FrameQueueObject, TelemetryQueueObject, CombinedFrametelemetryQueueObject
from typing import Optional

# ================================================================

logger = logging.getLogger("main.combiner")

if not logger.handlers:  # Avoid duplicate handlers
    video_handler = logging.FileHandler('/app/logs/combiner.log')
    video_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(video_handler)
    logger.setLevel(logging.DEBUG)

# ================================================================

class FrameTelemetryCombiner(mp.Process):
    """
    A multiprocessing Process that combines frames with telemetry data based on timestamps.
    
    Takes frames from frame_queue and telemetry dictionaries from telemetry_queue,
    matches them by timestamp, and outputs combined data to output_queues.
    Discards older telemetry data once a match is found.
    """
    
    def __init__(
            self, 
            frame_queue: mp.Queue, 
            telemetry_queue: mp.Queue, 
            output_queues: list[mp.Queue], 
            max_time_diff_s: float = 0.15
        ):
        """
        Initialize the FrameTelemetryCombiner.
        
        Args:
            frame_queue: Queue containing frame data with timestamps
            telemetry_queue: Queue containing telemetry dictionaries with timestamps
            output_queues: List of Queues to output combined frame-telemetry data to
            max_time_diff_s: Maximum acceptable time difference for matching (seconds)
        """
        super().__init__()
        self.frame_queue = frame_queue
        self.telemetry_queue = telemetry_queue
        self.output_queues = output_queues
        self.max_time_diff_s = max_time_diff_s
        self.telemetry_buffer = deque(frame_queue._maxsize)
        self.running = True
        
    def stop(self):
        """Signal the process to stop."""
        self.running = False
    
    def _load_telemetry_buffer(self):
        """Load available telemetry data into a DEQUEUE buffer."""
        while not self.telemetry_queue.empty():
            try:
                telemetry_data = self.telemetry_queue.get_nowait()
                self.telemetry_buffer.append(telemetry_data)
            except:
                logger.error("Error in retrieving objects from the telemetry queue")
                break
    
    def _find_best_telemetry_match(self, frame_timestamp: float) -> TelemetryQueueObject | None:
        """
        Find the best matching telemetry for the given frame timestamp.
        
        Returns the telemetry data with the closest timestamp within max_time_diff_s.
        Discards older telemetry data once a match is found.
        """
        if not self.telemetry_buffer:
            return None
        
        best_match = None
        best_time_diff = float('inf')
        match_index = -1
        
        # Find the best match
        for i, telemetry_data in enumerate(self.telemetry_buffer):
            try:
                time_diff = abs(frame_timestamp - telemetry_data.timestamp)
                
                if time_diff <= self.max_time_diff_s and time_diff < best_time_diff:
                    best_match = telemetry_data
                    best_time_diff = time_diff
                    match_index = i

                if best_match is not None and time_diff > self.max_time_diff_s:
                    # do not scan the entire data structure if a best match has already been found an now the time distance is growing
                    break

            except ValueError:
                logger.warning("expection rasied while searching for the best matching telemetry, continuing ...")
                continue
        
        # If we found a match, remove all telemetry data up to (but not including) the match.
        # The match is stored in 'best_match'
        if best_match is not None:
            logger.debug(f"best match found at index {match_index} out of {len(self.telemetry_buffer)} telemetries")
            for _ in range(match_index):
                if self.telemetry_buffer:
                    self.telemetry_buffer.popleft()
        
        return best_match
    
    def _combine_frame_telemetry(self, frame_data: FrameQueueObject, telemetry_data: Optional[TelemetryQueueObject]) -> CombinedFrametelemetryQueueObject:
        """
        Combine frame and telemetry data into output format.
        
        Returns:
            CombinedQueueObject object
        """

        combined_data = CombinedFrametelemetryQueueObject(
            frame_id = frame_data.frame_id, 
            frame=frame_data.frame, 
            telemetry=telemetry_data.telemetry if telemetry_data is not None else None,
           timestamp= frame_data.timestamp,
        )

        return combined_data
    
    def run(self):
        """Main process loop."""
        logger.info(f"FrameTelemetryCombiner process started (PID: {self.pid})")

        failed_matchings = 0
        
        while self.running:
            
            try:
                # Load new telemetry data into buffer
                self._load_telemetry_buffer()
                
                # Get frame data
                try:
                    frame_data = self.frame_queue.get(timeout=0.1)
                    if frame_data is None:  # Sentinel value to stop
                        logger.info("Found sentinel value on frame queue, terminating process ...")
                        break
                except Exception as e:
                    logger.error(f"Failed to retrieve frame from input queue: {e}")
                    continue
                
                # Extract frame timestamp
                frame_timestamp = frame_data.timestamp
                
                # Find best matching telemetry
                best_telemetry = self._find_best_telemetry_match(frame_timestamp)

                if best_telemetry is None:
                    failed_matchings += 1
                    logger.warning(
                        f"Unable to find suitable telemetry for frame {frame_timestamp.frame_id}. "
                        f"Total matches not found: {failed_matchings}"
                )
                
                # Combine frame and telemetry data
                combined_data = self._combine_frame_telemetry(frame_data, best_telemetry)
                
                # Output the combined frame and telemetry data
                try:
                    for output_queue in self.output_queues:        
                        output_queue.put(combined_data, timeout=1.0)
                except Exception as e:
                    logger.error(f"Failed to put combined frame-telemetry data onto one or more output queues: {e}")
                
                # Log matching info
                if best_telemetry:
                    telemetry_ts = best_telemetry.timestamp
                    time_diff = abs(frame_timestamp - telemetry_ts)
                    logger.debug(f"Matched frame {frame_timestamp:.3f} with telemetry {telemetry_ts:.3f} (diff: {time_diff:.3f}s)")
                else:
                    logger.warning(f"No telemetry match found for frame {frame_timestamp:.3f}")
                
            except Exception as e:
                logger.error(f"Error in FrameTelemetryCombiner: {e}")
                continue
        
        try:        
            for output_queue in self.output_queues:
                output_queue.put(None, timeout=1.0)
        except Exception as e:
                logger.error(f"Failed to put termination signal in one or more queues: {e}")
        
        logger.info("FrameTelemetryCombiner process finished")

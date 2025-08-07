import multiprocessing as mp
import cv2
import time
import logging
from pathlib import Path
import queue
from typing import Optional
from src.shared.processes.messages import AnnotationResults
from src.shared.processes.rtsp_streamer import RTSPStreamer

# ================================================================

logger = logging.getLogger("main.video_out")

if not logger.handlers:  # Avoid duplicate handlers
    video_handler = logging.FileHandler('/app/logs/video_out.log')
    video_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(video_handler)
    logger.setLevel(logging.DEBUG)

# ================================================================

class VideoStreamWriter(mp.Process):
    def __init__(
            self, 
            video_info_dict, 
            input_queue: mp.Queue, 
            output_dir: str, 
            output_url: str, 
            max_queue_size: int = 60, 
        ):

        super().__init__()
        self.video_info_dict = video_info_dict
        self.input_queue = input_queue
        self.output_dir = Path(output_dir)
        self.output_url = output_url
        self.max_queue_size = max_queue_size
        self.frames_dropped = 0
        self.frame_times = []  # Store timestamps for FPS calculation
    
    def _validate_output_url(self) -> bool:
        """Basic validation of output URL format."""
        if not self.output_url:
            logger.error("Empty output stream URL provided")
            return False
        
        # Check for common streaming protocols
        valid_protocols = ['rtmp://', 'rtsp://']
        if not any(self.output_url.startswith(protocol) for protocol in valid_protocols):
            logger.error(f"Output URL is not be a supported streaming URL: {self.output_url}")
            return False
        
        return True
    
    def _create_video_writers(self) -> tuple[Optional[cv2.VideoWriter], Optional[RTSPStreamer]]:
        """Create and validate video writers for file and stream output."""
            
        if not self._validate_output_url():
            return None, None
        
        videowriter_args = {
            "fourcc": cv2.VideoWriter_fourcc(*"mp4v"),
            "fps": int(self.video_info_dict["fps"]),
            "frameSize": (int(self.video_info_dict["frame_width"]), int(self.video_info_dict["frame_height"])),
        }
        
        out_file = None
        out_stream = None
        
        try:
            # Create file writer
            file_path = str(self.output_dir / "annotated_stream.mp4")
            logger.info(f"Logging to {file_path}")
            out_file = cv2.VideoWriter(file_path, **videowriter_args)
            
            if not out_file.isOpened():
                logger.error(f"Failed to open output file writer: {file_path}")
                out_file = None
            else:
                logger.info(f"Successfully opened file writer: {file_path}")
            
            # Create stream writer
            logger.info(f"streaming to {self.output_url}")
            out_stream = RTSPStreamer(self.output_url, int(self.video_info_dict["frame_width"]), int(self.video_info_dict["frame_height"]), int(self.video_info_dict["fps"]))
            out_stream.start()
            
            if not out_stream.ffmpeg_process:
                logger.error(f"Failed to start FFMPEG process")
                out_stream = None
            elif not out_stream.writer_thread:
                logger.error(f"Failed to start writer thread")
                out_stream = None
            else:
                logger.info(f"Successfully opened stream writer: {self.output_url}")
                
        except Exception as e:
            logger.error(f"Exception while creating video writers: {e}")
            if out_file:
                out_file.release()
            if out_stream:
                out_stream.stop()
            return None, None
        
        return out_file, out_stream
    
    def _manage_queue_buffer(self) -> None:
        """Drop frames if queue is getting too large to prevent excessive buffering."""
        try:
            queue_size = self.input_queue.qsize()
            if queue_size > self.max_queue_size:
                frames_to_drop = queue_size - self.max_queue_size
                logger.warning(f"Queue size ({queue_size}) exceeds maximum ({self.max_queue_size}). Dropping {frames_to_drop} frames.")
                
                for _ in range(frames_to_drop):
                    try:
                        dropped_frame = self.input_queue.get_nowait()
                        if dropped_frame is None:
                            # Put the sentinel back if we accidentally consumed it
                            self.input_queue.put(None)
                            break
                        self.frames_dropped += 1
                    except queue.Empty:
                        break
                        
                if self.frames_dropped > 0:
                    logger.warning(f"Total frames dropped due to buffering: {self.frames_dropped}")
                    
        except NotImplementedError:
            # qsize() not available on all platforms
            pass
    
    def _calculate_and_log_fps(self, frame_id: int) -> None:
        """Calculate and log FPS based on recent frame timestamps."""
        self.frame_times.append(time.time())
        if len(self.frame_times) > 30:  # Use last 30 frames to calculate FPS
            self.frame_times.pop(0)
            
        if len(self.frame_times) > 1:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            if time_diff > 0:
                avg_fps = len(self.frame_times) / time_diff
                logger.debug(f"Stream VIDEO frame {frame_id} | FPS: {avg_fps:.2f}")
            else:
                logger.debug(f"Stream VIDEO frame {frame_id}")
        else:
            logger.debug(f"Stream VIDEO frame {frame_id}")
    
    def run(self):
        """Main process loop for video streaming."""
        logger.info("Starting video streaming process")
        
        # Create output directory
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory created/verified: {self.output_dir}")
        except Exception as e:
            logger.error(f"Failed to create output directory {self.output_dir}: {e}")
            return
        
        # Create video writers
        out_file, out_stream = self._create_video_writers()
        
        if not out_file and not out_stream:
            logger.error("Failed to create any video writers. Terminating process.")
            return
        
        if not out_file:
            logger.warning("File writer not available. Only streaming will work.")
        
        if not out_stream:
            logger.warning("Stream writer not available. Only file output will work.")
        
        frames_processed = 0
        
        try:
            while True:
                # Manage queue buffer to prevent excessive memory usage
                self._manage_queue_buffer()
                
                # Get frame from queue with timeout
                try:
                    annotation_results: AnnotationResults  = self.input_queue.get(timeout=1.0)
                except queue.Empty:
                    logger.warning(f"No frame received within 1.0 seconds. Continuing to wait...")
                    continue
                
                # Check for termination sentinel
                if annotation_results is None:
                    logger.info("Received termination signal. Shutting down video streaming process.")
                    break

                frame_id = annotation_results.frame_id
                annotated_frame = annotation_results.annotated_frame 
                
                expected_height = int(self.video_info_dict["frame_height"])
                expected_width = int(self.video_info_dict["frame_width"])
                
                if annotated_frame.shape[:2] != (expected_height, expected_width):
                    logger.warning(f"Frame dimensions mismatch. Expected: ({expected_height}, {expected_width}), "
                                 f"Got: {annotated_frame.shape[:2]}")
                
                # Write frame to file
                if out_file:
                    try:
                        out_file.write(annotated_frame)
                    except Exception as e:
                        logger.error(f"Error writing frame {frame_id} to file: {e}")
                
                # Write frame to stream
                if out_stream:
                    try:
                        out_stream.write_frame(annotated_frame)
                    except Exception as e:
                        logger.error(f"Error writing frame {frame_id} to stream: {e}")
                
                # Track FPS and log progress
                self._calculate_and_log_fps(frame_id)
                frames_processed += 1
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt. Shutting down video streaming process.")
        except Exception as e:
            logger.error(f"Unexpected error in video streaming process: {e}")
        finally:
            # Clean up resources
            if out_file:
                try:
                    out_file.release()
                    logger.info("File writer released successfully")
                except Exception as e:
                    logger.error(f"Error releasing file writer: {e}")
            
            if out_stream:
                try:
                    out_stream.stop()
                    logger.info("Stream writer released successfully")
                except Exception as e:
                    logger.error(f"Error releasing stream writer: {e}")
            
            logger.info(f"Video streaming process terminated. Total frames processed: {frames_processed}, "
                       f"Frames dropped: {self.frames_dropped}")

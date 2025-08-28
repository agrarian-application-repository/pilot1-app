import multiprocessing as mp
import numpy as np
import cv2
import time
import logging
from pathlib import Path
from urllib.parse import urlparse
import socket
import msgpack
import queue
from typing import Optional
import struct
import datetime
from src.shared.processes.messages import AnnotationResults


# ================================================================

logger = logging.getLogger("main.alert_out")

if not logger.handlers:  # Avoid duplicate handlers
    video_handler = logging.FileHandler('/app/logs/alert_out.log', mode='w')
    video_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(video_handler)
    logger.setLevel(logging.DEBUG)

# ================================================================


class NotificationsStreamWriter(mp.Process):

    def __init__(
            self, 
            video_info_dict, 
            cooldown_seconds: float, 
            input_queue: mp.Queue, 
            output_dir: str, 
            output_url: str,

            max_queue_size: int = 30,
            jpeg_quality: int = 85, 
            max_packet_size: int = 65507,
            connection_timeout: float = 10.0, 
            retry_attempts: int = 3,
            retry_delay: float = 1.0
    ):
        super().__init__()
        self.video_info_dict = video_info_dict
        self.input_queue = input_queue
        self.output_dir = Path(output_dir)
        self.cooldown_seconds = cooldown_seconds
        
        self.max_queue_size = max_queue_size
        self.jpeg_quality = max(1, min(100, jpeg_quality))
        self.max_packet_size = max_packet_size
        self.connection_timeout = connection_timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        # Calculate cooldown in frames
        fps = self.video_info_dict["fps"]
        self.last_alert_frame_id = -fps
        self.alerts_frames_cooldown = fps * cooldown_seconds
        
        # Parse and validate URL
        self.protocol, self.host, self.port = self._parse_and_validate_url(output_url)
        
        # Initialize network resources
        self.sock = None
        self.is_connected = False
        self.alerts_sent = 0
        self.alerts_failed = 0
        self.frames_dropped = 0
        
        # File handle initialized in run()
        self.alerts_file = None
    
    def _parse_and_validate_url(self, output_url: str) -> tuple[str, str, int]:
        """Parse and validate the output URL."""
        try:
            parsed_url = urlparse(output_url)
            protocol = parsed_url.scheme.lower()
            
            if protocol != 'tcp':
                logger.error(f"Invalid protocol: {protocol}. Expected 'tcp'.")
                raise ValueError(f"Unsupported protocol: {protocol}")
            
            if not parsed_url.hostname:
                logger.error("Missing hostname in URL")
                raise ValueError("Missing hostname in URL")
            
            if not parsed_url.port:
                logger.error("Missing port in URL")
                raise ValueError("Missing port in URL")
            
            host = parsed_url.hostname
            port = parsed_url.port
            
            # Validate port range
            if not (1 <= port <= 65535):
                logger.error(f"Invalid port number: {port}")
                raise ValueError(f"Invalid port number: {port}")
            
            logger.info(f"Parsed URL - Protocol: {protocol}, Host: {host}, Port: {port}")
            return protocol, host, port
            
        except Exception as e:
            logger.error(f"Failed to parse URL {output_url}: {e}")
            raise
    
    def _create_socket(self) -> Optional[socket.socket]:
        """Create and configure TCP socket."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.settimeout(self.connection_timeout)
            logger.info("Created TCP socket")
            return sock
        except Exception as e:
            logger.error(f"Failed to create socket: {e}")
            return None
    
    def _connect_tcp(self) -> bool:
        """Establish TCP connection with retry logic."""
        
        for attempt in range(1, self.retry_attempts+1):
            try:
                if self.sock:
                    self.sock.close()
                
                self.sock = self._create_socket()
                if not self.sock:
                    continue
                
                logger.info(f"Attempting TCP connection to {self.host}:{self.port} (attempt {attempt}/{self.retry_attempts})")
                self.sock.connect((self.host, self.port))
                self.is_connected = True
                logger.info(f"Successfully connected to {self.host}:{self.port}")
                return True
                
            except socket.timeout:
                logger.warning(f"TCP connection timeout (attempt {attempt}/{self.retry_attempts})")
            except socket.error as e:
                logger.warning(f"TCP connection failed (attempt {attempt}/{self.retry_attempts}): {e}")
            except Exception as e:
                logger.error(f"Unexpected error during TCP connection (attempt {attempt}/{self.retry_attempts}): {e}")
            
            if attempt < self.retry_attempts:
                logger.info(f"Retrying to connect in {self.retry_delay} seconds ...")
                time.sleep(self.retry_delay)
        
        logger.error(f"Failed to establish TCP connection after {self.retry_attempts} attempts")
        return False
    
    def _manage_queue_buffer(self) -> None:
        """Drop frames if queue is getting too large."""
        try:
            queue_size = self.input_queue.qsize()
            if queue_size > self.max_queue_size:
                frames_to_drop = queue_size - self.max_queue_size
                logger.warning(f"Queue size ({queue_size}) exceeds maximum ({self.max_queue_size}). Dropping {frames_to_drop} frames.")
                
                for _ in range(frames_to_drop):
                    try:
                        dropped_frame = self.input_queue.get_nowait()
                        if dropped_frame is None:
                            self.input_queue.put(None)  # Put sentinel back
                            break
                        self.frames_dropped += 1
                    except queue.Empty:
                        break
                        
        except NotImplementedError:
            # qsize() not available on all platforms
            pass
    
    def _compress_image(self, frame: np.ndarray) -> tuple[bool, np.ndarray, Optional[str]]:
        """Compress image to reduce network transfer size."""
        try:
            # Validate frame
            if frame is None or frame.size == 0:
                logger.error("Invalid frame for compression")
                return False, frame, None
            
            # Compress to JPEG
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            success, compressed_img = cv2.imencode(".jpg", frame, encode_params)
            
            if not success:
                logger.warning("Image compression failed, using original frame")
                return False, frame, None
            
            # Log compression ratio
            original_size = frame.nbytes
            compressed_size = len(compressed_img)
            compression_ratio = original_size / compressed_size if compressed_size > 0.0 else 1.0
            logger.debug(f"Image compressed: {original_size} -> {compressed_size} bytes (ratio: {compression_ratio:.2f})")
            
            return True, compressed_img, '.jpg'
            
        except Exception as e:
            logger.error(f"Error during image compression: {e}")
            return False, frame, None
    
    def _create_message_packet(self, frame_results: AnnotationResults) -> Optional[bytes]:
        """Create msgpack data packet from frame results."""
        try:
            # Compress image
            compression_success, compressed_img, img_format = self._compress_image(frame_results.annotated_frame)
            
            # Create message data
            message_data = {
                "frame_id": frame_results.frame_id,
                "alert_msg": frame_results.alert_msg,
                "timestamp": frame_results.timestamp,
                "img_shape": frame_results.annotated_frame.shape,
                "img_dtype": str(frame_results.annotated_frame.dtype),
                "img_format": img_format,
                "compressed": compression_success
            }
            
            # Add image data
            if compression_success and img_format:
                message_data["img_bytes"] = compressed_img.tobytes()
            else:
                message_data["img_bytes"] = frame_results.annotated_frame.tobytes()
            
            # Pack with msgpack
            msgpack_data = msgpack.packb(message_data, use_bin_type=True)
            
            if msgpack_data:
                # Check packet size
                if len(msgpack_data) > self.max_packet_size:
                    logger.warning(
                        f"Packet size ({len(msgpack_data)}) exceeds maximum ({self.max_packet_size}). "
                        f"Consider reducing image quality or resolution."
                    )
                
                return msgpack_data
            
            else:
                logger.error(f"Error creating message packet: {e}")
                return None

        except Exception as e:
            logger.error(f"Error creating message packet: {e}")
            return None

    def _send_all(self, data: bytes) -> bool:
        """Send all data, handling partial sends"""
        total_sent = 0
        while total_sent < len(data):
            try:
                sent = self.sock.send(data[total_sent:])
                if sent == 0:
                    logger.warning("sent nothing")
                    return False
                total_sent += sent
            except socket.error:
                logger.warning("socket error in send")
                return False
            
        return True

    def _send_packet(self, data: bytes) -> bool:
        """Send data via TCP with length prefix."""
        try:
            if not self.sock or not self.is_connected:
                logger.warning("TCP socket not connected, attempting reconnection ...")
                logger.debug(f"socket: {self.sock}, connected: {self.is_connected} ")
                if not self._connect_tcp():
                    return False
            
            # Send length prefix (4 bytes) followed by data
            length_prefix = struct.pack('!I', len(data))
            full_packet = length_prefix + data
            
            # Use _send_all to handle partial sends properly
            if not self._send_all(full_packet):
                logger.warning("Failed to send complete packet, disconnecting ...")
                self.is_connected = False
                return False
                
            logger.info("TCP packet sent succesfully")
            return True
        
        except socket.timeout:
            logger.error("TCP send timeout")
            self.is_connected = False
            return False
        except socket.error as e:
            logger.error(f"TCP send error: {e}")
            self.is_connected = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error during TCP send: {e}")
            self.is_connected = False
            return False
    
    def _write_alert_to_file(self, frame_results: AnnotationResults) -> None:
        """Write alert message to file."""
        try:
            if not self.alerts_file:
                logger.error("Alerts file not open")
                return
            
            msg = f"Frame {frame_results.frame_id}, "
            msg += f"Alert: {frame_results.alert_msg}, "
            msg += f"timestamp: {datetime.datetime.fromtimestamp(frame_results.timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')}\n"
            
            self.alerts_file.write(msg)
            self.alerts_file.flush()  # Ensure data is written immediately
            
        except Exception as e:
            logger.error(f"Error writing alert to file: {e}")

    def _cleanup(self):
        if self.alerts_file:
            self.alerts_file.close()
        # Cleanup resources
        self._close_alerts_file()
        self._close_socket()

    def _close_alerts_file(self):
        if self.alerts_file:
            try:
                self.alerts_file.close()
                logger.info("Alerts file closed")
            except Exception as e:
                logger.error(f"Error closing alerts file: {e}")

    def _close_socket(self):
        if self.sock:
            try:
                self.sock.close()
                logger.info("Socket closed")
            except Exception as e:
                logger.error(f"Error closing socket: {e}")

    def run(self):
        """Main process loop for notifications streaming."""
        logger.info("Starting notification streaming process")
        
        # Create output directory
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory created/verified: {self.output_dir}")
        except Exception as e:
            logger.error(f"Failed to create output directory {self.output_dir}: {e}")
            return
        
        # Open alerts file
        try:
            alerts_file_path = self.output_dir / "alerts.txt"
            self.alerts_file = open(alerts_file_path, "w")
            logger.info(f"Alerts file opened: {alerts_file_path}")
        except Exception as e:
            logger.error(f"Failed to open alerts file: {e}")
        
        # create TCP socket and connect
        connection_established = self._connect_tcp()
        if not connection_established:
            logger.warning("Failed to establish TCP connection, will try to reconnet on send")
        
        frames_processed = 0
        
        try:
            while True:
                
                # Manage queue buffer
                self._manage_queue_buffer()
                
                # Get frame from queue with timeout
                try:
                    frame_results: AnnotationResults = self.input_queue.get(timeout=1.0)
                except queue.Empty:
                    logger.warning(f"No frame received within 1.0 seconds. Continuing to wait...")
                    continue
                
                # Check for termination sentinel
                if frame_results is None:
                    logger.info("Received termination signal. Shutting down notification streaming process.")
                    break
                
                frame_id = frame_results.frame_id
                frames_processed += 1
                
                # Check if alert should be sent
                cooldown_has_passed = (frame_id - self.last_alert_frame_id) >= self.alerts_frames_cooldown
                alert_exist = len(frame_results.alert_msg) > 0
                
                if cooldown_has_passed and alert_exist:

                    # Write to file (if open)
                    if self.alerts_file:
                        self._write_alert_to_file(frame_results)

                    # Send alert via TCP (if not connected, send will try to reconnect)
                    
                    # Create message packet
                    msgpack_data = self._create_message_packet(frame_results)
                    if not msgpack_data:
                        logger.error("Failed to create message packet")
                        continue
                    
                    # Send packet
                    if self._send_packet(msgpack_data):
                        self.alerts_sent += 1
                        logger.info(
                            f"Alert sent successfully (Frame ID: {frame_id}), "
                            f"Packet size: {len(msgpack_data)} bytes, "
                            f"Alert: {frame_results.alert_msg})"
                        )
                    else:
                        self.alerts_failed += 1
                        logger.error(f"Failed to send alert (Frame ID: {frame_id})")
                    
                    # Update last alert frame ID to reset cooldown
                    self.last_alert_frame_id = frame_id
                
                # Log periodic status
                if frames_processed % 100 == 0:
                    logger.info(
                        f"Processed {frames_processed} frames, "
                        f"Alerts sent: {self.alerts_sent}, "
                        f"Alerts failed: {self.alerts_failed}, "
                        f"Frames dropped: {self.frames_dropped}"
                    )

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt. Shutting down notification streaming process.")
        except Exception as e:
            logger.error(f"Unexpected error in notification streaming process: {e}")
        
        finally:
            self._cleanup()
            logger.info(
                f"Notification streaming process terminated. "
                f"Frames processed: {frames_processed}, "
                f"Alerts sent: {self.alerts_sent}, "
                f"Alerts failed: {self.alerts_failed}, "
                f"Frames dropped: {self.frames_dropped}"
            )

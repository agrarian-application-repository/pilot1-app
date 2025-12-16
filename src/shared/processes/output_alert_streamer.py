import multiprocessing as mp
import json
import cv2
import numpy as np
from collections import deque
from typing import Optional
import base64
from datetime import datetime as dtt
import logging
from time import time
from src.shared.processes.db_manager import DatabaseManager
from src.shared.processes.websocket_manager import WebSocketManager
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
    """
    A multiprocessing process that:
    - receives AnnotationResults objects with anomalies
    - maintains recent alerts
    - exposes WebSocket API and pushes alerts to clients
    - saves alerts to a SQL database
    - logs alerts to file.
    """

    def __init__(
            self,
            input_queue: mp.Queue,
            stop_event: mp.Event,

            max_alerts: int = 100,
            use_websocket: bool = True,
            websocket_host: str = "localhost",
            websocket_port: int = 8765,
            log_file: Optional[str] = "alerts.log",
            jpeg_quality: int = 85,
            database_url: Optional[str] = None
    ):
        """
        Initialize the NotificationsStreamWriter process.

        Args:
            input_queue: Queue receiving AnnotationResults objects
            stop_event: Event to signal process termination
            max_alerts: Maximum number of recent alerts to maintain
            use_websocket: wheter to use websocket
            websocket_host: Host for WebSocket server
            websocket_port: Port for WebSocket server
            log_file: Path to log file (None to disable logging)
            jpeg_quality: JPEG compression quality (0-100)
            database_url: SQLAlchemy database URL
        """
        super().__init__()
        self.input_queue = input_queue
        self.stop_event = stop_event
        self.max_alerts = max_alerts
        self.log_file = log_file
        self.jpeg_quality = jpeg_quality

        # Recent alerts buffer
        self.recent_alerts = deque(maxlen=max_alerts)

        # Initialize DB manager
        self.db_manager = DatabaseManager(database_url) if database_url is not None else None

        # Initialize WebSocket manager
        self.ws_manager = WebSocketManager(websocket_host, websocket_port, stop_event) if use_websocket else None

    @staticmethod
    def _resize_frame(frame: np.ndarray, original_wh: tuple) -> np.ndarray:
        """
        Resize frame back to original dimensions.

        Args:
            frame: The processed frame
            original_wh: Tuple of (width, height)

        Returns:
            Resized frame
        """
        width, height = original_wh
        logger.debug(f"Resizing frame from {frame.shape[:2][::-1]} to {original_wh}")
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    def _compress_frame(self, frame: np.ndarray) -> tuple:
        """
        Compress frame to JPEG format.

        Args:
            frame: The frame to compress

        Returns:
            Tuple of (base64_encoded_string, raw_bytes)
        """
        compression_start = time()

        # Encode as JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)

        # Convert to base64 for WebSocket transmission
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        # Get raw bytes for database storage
        compressed_bytes = buffer.tobytes()

        compressed_size = len(buffer)
        logger.debug(
            f"Compressed frame to {compressed_size} bytes (quality={self.jpeg_quality}). "
            f"Took {(time() - compression_start) * 1000:.1f} ms"
        )

        return jpg_as_text, compressed_bytes

    def _log_alert(self, frame_id: int, alert_msg: str, timestamp: float, datetime: str):
        """
        Log alert information to file.

        Args:
            frame_id: Frame identifier
            alert_msg: Alert message
            timestamp: Alert timestamp
            datetime: alert datetime
        """
        try:
            with open(self.log_file, 'a') as f:
                log_entry = {
                    'frame_id': frame_id,
                    'alert_msg': alert_msg,
                    'timestamp': timestamp,
                    'datetime': datetime,
                }
                f.write(json.dumps(log_entry) + '\n')
            logger.debug(f"Alert logged to file: frame_id={frame_id}")
        except Exception as e:
            logger.error(f"Error logging alert to file: {e}")

    def _process_alert(self, alert: AnnotationResults):
        """
        Process a new alert: resize, compress, store, and queue for broadcast.

        Args:
            alert: AnnotationResults object
        """
        logger.info(f"Processing alert: frame_id={alert.frame_id}, msg='{alert.alert_msg}'")

        # Resize frame to original dimensions
        width, height = alert.original_wh
        resized_frame = self._resize_frame(alert.annotated_frame, alert.original_wh)

        # Compress frame
        compressed_frame, compressed_bytes = self._compress_frame(resized_frame)

        # Create alert data structure
        alert_datetime = dtt.fromtimestamp(alert.timestamp).isoformat()
        alert_data = {
            'frame_id': alert.frame_id,
            'alert_msg': alert.alert_msg,
            'timestamp': alert.timestamp,
            'datetime': alert_datetime,
            'image': compressed_frame,
            'width': width,
            'height': height,
            'compression': 'jpeg',
        }

        # Add to recent alerts (dequeue automatically remove old alerts when is full)
        self.recent_alerts.append(alert_data)
        logger.debug(f"Alert added to recent alerts buffer (size: {len(self.recent_alerts)})")

        # Log alert to file
        if self.log_file:
            self._log_alert(alert.frame_id, alert.alert_msg, alert.timestamp, alert_datetime)

        # Save to database
        if self.db_manager:
            self.db_manager.save_alert(
                frame_id=alert.frame_id,
                alert_msg=alert.alert_msg,
                timestamp=alert.timestamp,
                datetime=alert_datetime,
                image_data=compressed_bytes,
                image_width=width,
                image_height=height,
            )

        # Queue for WebSocket broadcast
        if self.ws_manager:
            self.ws_manager.queue_alert(alert_data)

    def run(self):
        """Main process loop."""
        ws_status = f"ws://{self.ws_manager.host}:{self.ws_manager.port}" if self.ws_manager else "disabled"
        db_status = self.db_manager.database_url if self.db_manager else "disabled"
        logfile_status = self.log_file if self.log_file else 'disabled'

        logger.info("=" * 60)
        logger.info("NotificationsStreamWriter process starting")
        logger.info(f"Configuration:")
        logger.info(f"  - Max alerts buffer: {self.max_alerts}")
        logger.info(f"  - WebSocket: {ws_status}")
        logger.info(f"  - Database: {db_status}")
        logger.info(f"  - Log file: {logfile_status}")
        logger.info(f"  - JPEG quality: {self.jpeg_quality}")
        logger.info("=" * 60)

        if self.db_manager:
            # Initialize database
            self.db_manager.initialize()

        if self.ws_manager:
            # Set recent alerts reference for WebSocket manager
            self.ws_manager.set_recent_alerts(self.recent_alerts)
            # Start WebSocket server
            self.ws_manager.start()

        alert_count = 0

        while not self.stop_event.is_set():

            try:
                # Get alert from queue with timeout
                alert = self.input_queue.get(timeout=0.1)
                # Process the alert
                self._process_alert(alert)
                alert_count += 1

            except mp.queues.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing alert: {e}", exc_info=True)

        logger.info("=" * 60)
        logger.info("NotificationsStreamWriter process terminating ...")
        logger.info(f"Total alerts processed: {alert_count}")

        # Close database
        if self.db_manager:
            self.db_manager.close()

        # Stop WebSocket server
        if self.ws_manager:
            self.ws_manager.stop()

        logger.info("NotificationsStreamWriter process terminated")
        logger.info("=" * 60)


"""
# Example usage
if __name__ == "__main__":
    # Create shared objects
    alert_queue = mp.Queue()
    stop_event = mp.Event()

    # Create and start process
    writer = NotificationsStreamWriter(
        input_queue=alert_queue,
        stop_event=stop_event,
        max_alerts=50,
        websocket_port=8765,
        log_file="alerts.log",
        jpeg_quality=85,
        database_url="sqlite:///alerts.db"  # Use SQLite for example
        # For PostgreSQL: "postgresql://user:password@localhost:5432/alerts_db"
        # For MySQL: "mysql+pymysql://user:password@localhost:3306/alerts_db"
    )
    writer.start()

    # Simulate sending alerts
    import time

    for i in range(5):
        fake_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        alert = AnnotationResults(
            frame_id=i,
            annotated_frame=fake_frame,
            alert_msg=f"Danger detected: Type {i}",
            timestamp=time.time(),
            original_wh=(1920, 1080)
        )
        alert_queue.put(alert)
        time.sleep(1)

    # Let it run for a bit
    time.sleep(5)

    # Stop the process
    stop_event.set()
    writer.join()
    print("Process terminated")
"""

import multiprocessing as mp
import json
import cv2
import threading
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
from src.shared.processes.constants import (
    WS_COMMON_PORT,
    WS_PORT,
    WSS_PORT,
    MAX_ALERTS_STORED,
    JPEG_COMPRESSION_QUALITY,
    ALERTS_GET_TIMEOUT,
    UPSAMPLING_MODE,
    POISON_PILL,
)

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
    - exposes WebSocket API and pushes alerts to clients (optional)
    - saves alerts to a SQL database (optional)
    - logs alerts to file. (optional)
    """

    def __init__(
            self,
            input_queue: mp.Queue,
            error_event: mp.Event,
            max_alerts: int = MAX_ALERTS_STORED,
            websocket_host: Optional[str] = "localhost",
            websocket_port: int = WSS_PORT,
            log_file_path: Optional[str] = "alerts.log",
            jpeg_quality: int = JPEG_COMPRESSION_QUALITY,
            database_url: Optional[str] = None,
            alerts_get_timeout: float = ALERTS_GET_TIMEOUT,
    ):
        """
        Initialize the NotificationsStreamWriter process.

        Args:
            input_queue: Queue receiving AnnotationResults objects
            error_event: Event to signal process termination, processing error
            websocket_host: Host for WebSocket server
            websocket_port: Port for WebSocket server
            log_file: Path to log file (None to disable logging)
            jpeg_quality: JPEG compression quality (0-100)
            database_url: SQLAlchemy database URL
        """
        super().__init__()
        self.input_queue = input_queue
        self.error_event = error_event
        self.log_file_path = log_file_path
        self.jpeg_quality = jpeg_quality
        self.alerts_get_timeout = alerts_get_timeout

        # Initialize log file (placeholder, instantiated in run)
        self.log_file = None

        # Initialize DB manager (placeholder, instantiated in run)
        self.database_url = database_url
        self.db_manager = None

        # Initialize WebSocket manager (placeholder, instantiated in run)
        self.websocket_host = websocket_host
        self.websocket_port = websocket_port
        self.ws_manager = None

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
        current_height, current_width, _ = frame.shape
        width, height = original_wh
        logger.debug(f"Resizing frame from {({current_width},{current_height})} to {(width, height)}")
        return cv2.resize(frame, (width, height), interpolation=UPSAMPLING_MODE)

    def _compress_frame(self, frame: np.ndarray) -> tuple:
        """
        Compress frame to JPEG format.

        Args:
            frame: The frame to compress

        Returns:
            Tuple of (base64_encoded_string, raw_bytes)
        """

        jpg_as_text = 'no_ws'
        compressed_bytes = 'no_db'

        if not self.ws_manager and not self.db_manager:
            logger.debug(f"No WS nor DB, skipping compression step.")
            return jpg_as_text, compressed_bytes

        compression_start = time()

        # Encode as JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)

        # Convert to base64 for WebSocket transmission
        if self.ws_manager:
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        # Get raw bytes for database storage
        if self.db_manager:
            compressed_bytes = buffer.tobytes()

        logger.debug(f"Frame compressed took frame to {(time() - compression_start) * 1000:.1f} ms")

        return jpg_as_text, compressed_bytes

    def _log_alert(self, frame_id: int, alert_msg: str, timestamp: float, datetime_str: str):
        """
        Log alert information using a persistent file handle.

        Args:
            frame_id: Frame identifier
            alert_msg: Alert message
            timestamp: Alert timestamp
            datetime: alert datetime
        """
        if not self.log_file:
            return

        try:
            log_entry = {
                'frame_id': frame_id,
                'alert_msg': alert_msg,
                'timestamp': timestamp,
                'datetime': datetime_str,
            }
            # Writing to a line-buffered handle is highly efficient
            self.log_file.write(json.dumps(log_entry) + '\n')
            logger.debug(f"Alert logged to file: frame_id={frame_id}")

        except Exception as e:
            logger.error(f"Error writing to persistent log: {e}")

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

        # Log alert to file using the handle
        if self.log_file:
            self._log_alert(alert.frame_id, alert.alert_msg, alert.timestamp, alert_datetime)

        # Queue for WebSocket broadcast
        if self.ws_manager:
            self.ws_manager.queue_alert(alert_data)

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

    def run(self):
        """Main process loop."""

        alert_count = 0
        poison_pill_received = False

        ws_status = f"ws://{self.websocket_host}:{self.websocket_port}" if self.ws_manager else "disabled"
        db_status = self.db_manager.database_url if self.db_manager else "disabled"
        logfile_status = self.log_file_path if self.log_file_path else 'disabled'

        logger.info("=" * 60)
        logger.info("NotificationsStreamWriter process starting")
        logger.info(f"Configuration:")
        logger.info(f"  - Max alerts buffer: {self.max_alerts}")
        logger.info(f"  - WebSocket: {ws_status}")
        logger.info(f"  - Database: {db_status}")
        logger.info(f"  - Log file: {logfile_status}")
        logger.info(f"  - JPEG quality: {self.jpeg_quality}")
        logger.info("=" * 60)

        # ---------------------------------
        # Instantiate output managers
        # ---------------------------------

        # Initialize log file manager
        try:
            self.log_file = open(self.log_file_path, 'a', buffering=1, encoding='utf-8') if self.log_file_path else None
        except Exception as e:
            logger.error(f"Failed to open log file {self.log_file_path}: {e}. Continuing anyway ...")

        # Initialize DB manager
        try:
            if self.database_url:
                # Instantiate object
                self.db_manager = DatabaseManager(database_url=self.database_url)
                # Initialize database
                self.db_manager.initialize()
            else:
                # db_manager remains None
                pass
        except Exception as e:
            self.db_manager = None  # ensure db_manager remains None stays None
            logger.error(f"Failed to create DB manager: {e}. Continuing ...")

        # Initialize WebSocket manager
        try:
            if self.websocket_host:
                # Instantiate object
                self.ws_manager = WebSocketManager(
                    host=self.websocket_host,
                    port=self.websocket_port,
                )
                # Start WebSocket server
                self.ws_manager.start()
            else:
                # ws_manager remains None
                pass
        except Exception as e:
            self.ws_manager = None  # ensure ws_manager stays None
            logger.error(f"Failed to create websocket server: {e}. Continuing ...")

        # At least one between Websocket manager and DB manager must have been initialized
        if not (self.db_manager or self.ws_manager):
            self.error_event.set()
            logger.error(
                "Error event set: "
                "No available system (WebSocket, DB) for providing alerts. "
                "Shutting down the application ..."
            )

        # ---------------------------------
        # Alerts Processing
        # ---------------------------------

        while not self.error_event.is_set():

            try:
                # Get alert from queue with timeout
                alert = self.input_queue.get(timeout=self.alerts_get_timeout)

                # if the alert being processed is the poison pill,
                # break out of the loop to complete shutdown the process
                if alert == POISON_PILL:
                    poison_pill_received = True
                    break

                # Process the alert
                self._process_alert(alert)
                alert_count += 1

            except mp.queues.Empty:
                logger.info("Input queue empty. Continuing to wait for alerts ... ")
                continue
            except Exception as e:
                logger.error(f"Error processing alert: {e}. Will try to process next ones ...", exc_info=True)
                continue

        # ---------------------------------
        # Final cleanup
        # ---------------------------------

        logger.info("=" * 60)
        logger.info("NotificationsStreamWriter process terminating ...")
        logger.info(f"Total alerts processed: {alert_count}")

        # Close log file
        if self.log_file:
            logger.info("Closing alert log file...")
            self.log_file.close()
            logger.info("Alert log file closed")

        # Close database
        if self.db_manager:
            self.db_manager.close()

        # Stop WebSocket server
        if self.ws_manager:
            self.ws_manager.stop()

        logger.info(
            f"NotificationsStreamWriter process terminated. "
            f"Poison Pill: {poison_pill_received}. "
            f"Error event: {self.error_event.is_set()}."
        )
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

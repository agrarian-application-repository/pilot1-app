import multiprocessing as mp
from queue import Empty as QueueEmptyException
import json
import cv2
import numpy as np
from typing import Optional
import base64
from datetime import datetime as dtt
import logging
from time import time
from src.shared.processes.db_manager import DatabaseManager
from src.shared.processes.websocket_manager import WebSocketManager
from src.shared.processes.messages import AnnotationResults
from src.shared.processes.constants import *


# ================================================================

logger = logging.getLogger("main.alert_out")

if not logger.handlers:  # Avoid duplicate handlers
    video_handler = logging.FileHandler('./logs/alert_out.log', mode='w')
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
            alerts_get_timeout: float = ALERTS_QUEUE_GET_TIMEOUT,
            alerts_max_consecutive_failures: int = ALERTS_MAX_CONSECUTIVE_FAILURES,
            alerts_jpeg_quality: int = ALERTS_JPEG_COMPRESSION_QUALITY,
            # ------- FILE manager parameters --------
            log_file_path: Optional[str] = "alerts.log",
            # ------- WS manager parameters --------
            websocket_host: Optional[str] = None,
            websocket_port: int = WSS_PORT,
            ws_manager_ping_interval: float = WS_MANAGER_PING_INTERVAL,
            ws_manager_ping_timeout: float = WS_MANAGER_PING_TIMEOUT,
            ws_manager_broadcast_timeout: float = WS_MANAGER_BROADCAST_TIMEOUT,
            ws_manager_thread_close_timeout: float = WS_MANAGER_THREAD_CLOSE_TIMEOUT,
            # ------- DB manager parameters --------
            database_service: Optional[str] = None,
            database_host: Optional[str] = None,
            database_port: int = DB_PORT,
            database_username: str = "",
            database_password: str = "",
            db_manager_pool_size: int = DB_MANAGER_POOL_SIZE,
            db_manager_max_overflow: int = DB_MANAGER_MAX_OVERFLOW,
            db_manager_queue_get_timeout: float = DB_MANAGER_QUEUE_WAIT_TIMEOUT,
            db_manager_thread_close_timeout: float = DB_MANAGER_THREAD_CLOSE_TIMEOUT,
            db_manager_alerts_queue_size: int = DB_MANAGER_QUEUE_SIZE,

    ):
        """
        Initialize the NotificationsStreamWriter process.

        Args:
            input_queue: Queue receiving AnnotationResults objects
            error_event: Event to signal process termination, processing error
            websocket_host: Host for WebSocket server
            websocket_port: Port for WebSocket server
            log_file: Path to log file (None to disable logging)
            alerts_jpeg_quality: JPEG compression quality (0-100)
            database_url: SQLAlchemy database URL


        """
        super().__init__()
        self.input_queue = input_queue
        self.error_event = error_event
        self.log_file_path = log_file_path
        self.alerts_jpeg_quality = alerts_jpeg_quality

        self.alerts_get_timeout = alerts_get_timeout
        self.alerts_max_consecutive_failures = alerts_max_consecutive_failures

        # Initialize log file (placeholder, instantiated in run)
        self.log_file = None

        # initialize database url
        if database_username and database_password:
            auth = f"{database_username}:{database_password}@"
        elif database_username:
            auth = f"{database_username}@"
        else:
            auth = ""

        if database_service == POSTGRESQL:
            self.database_url = f"postgresql://{auth}{database_host}:{database_port}/{DB_NAME}"
        elif database_service == MYSQL:
            self.database_url = f"mysql+pymysql://{auth}{database_host}:{database_port}/{DB_NAME}"
        elif database_service == SQLITE:
            self.database_url = f"sqlite:///{DB_NAME}"
        else:
            self.database_url = None

        
        self.db_username = database_username
        self.db_password = database_password
        self.db_manager = None

        # Initialize WebSocket manager (placeholder, instantiated in run)
        self.websocket_host = websocket_host
        self.websocket_port = websocket_port
        self.ws_manager = None

        # WS Manager Config
        self.ws_manager_ping_interval = ws_manager_ping_interval
        self.ws_manager_ping_timeout = ws_manager_ping_timeout
        self.ws_manager_broadcast_timeout = ws_manager_broadcast_timeout
        self.ws_manager_thread_close_timeout = ws_manager_thread_close_timeout

        # DB manager Config
        self.db_manager_pool_size = db_manager_pool_size
        self.db_manager_max_overflow = db_manager_max_overflow
        self.db_manager_queue_get_timeout = db_manager_queue_get_timeout
        self.db_manager_thread_close_timeout = db_manager_thread_close_timeout
        self.db_manager_alerts_queue_size = db_manager_alerts_queue_size

        self.work_finished = mp.Event()

    def _setup_managers(self):

        # Initialize log file manager
        try:
            if self.log_file_path:
                self.log_file = open(self.log_file_path, 'a', buffering=1, encoding='utf-8')
            else:
                # log_file remains None
                pass
        except Exception as e:
            self.log_file = None    # ensure log_file stays None
            logger.error(f"Failed to open log file {self.log_file_path}: {e}. Continuing without ...")

        # Initialize DB manager
        try:
            if self.database_url:
                # Instantiate object
                self.db_manager = DatabaseManager(
                    database_url=self.database_url,
                    alerts_queue_size=self.db_manager_alerts_queue_size,
                    pool_size=self.db_manager_pool_size,
                    max_overflow=self.db_manager_max_overflow,
                    queue_get_timeout=self.db_manager_queue_get_timeout,
                    thread_close_timeout=self.db_manager_thread_close_timeout,
                )
                # Initialize database
                # Raises exception on failure
                self.db_manager.initialize(self.db_username, self.db_password)
            else:
                # db_manager remains None
                pass
        except Exception as e:
            self.db_manager = None  # ensure db_manager stays None
            logger.error(f"Failed to create DB manager: {e}. Continuing without ...")

        # Initialize WebSocket manager
        try:
            if self.websocket_host:
                # Instantiate object
                self.ws_manager = WebSocketManager(
                    host=self.websocket_host,
                    port=self.websocket_port,
                    ping_interval=self.ws_manager_ping_interval,
                    ping_timeout=self.ws_manager_ping_timeout,
                    broadcast_timeout=self.ws_manager_broadcast_timeout,
                    thread_close_timeout=self.ws_manager_thread_close_timeout,
                )
                # Start WebSocket server
                self.ws_manager.start()
            else:
                # ws_manager remains None
                pass
        except Exception as e:
            self.ws_manager = None  # ensure ws_manager stays None
            logger.error(f"Failed to create websocket server: {e}. Continuing without ...")

        # At least one between Websocket manager and DB manager must have been initialized
        if not (self.db_manager or self.ws_manager or self.log_file):
            self.error_event.set()
            logger.error(
                "Error event set: "
                "No available system (File, WebSocket, DB) for providing alerts. "
                "Shutting down the application ..."
            )
            raise RuntimeError("No output managers available")
            # on failure, main will catch exception and cause termination

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

        if not (self.ws_manager or self.db_manager):
            logger.debug(f"No WS nor DB, skipping compression step.")
            return jpg_as_text, compressed_bytes

        compression_start = time()

        # Encode as JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.alerts_jpeg_quality]
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
            datetime_str: alert datetime
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
        Process a new alert: compress, store, and queue for broadcast.

        Args:
            alert: AnnotationResults object
        """
        logger.info(f"Processing alert: frame_id={alert.frame_id}, msg='{alert.alert_msg}'")

        # Compress frame
        compressed_frame, compressed_bytes = self._compress_frame(alert.annotated_frame)

        # Create alert data structure
        
        alert_datetime = dtt.fromtimestamp(alert.timestamp)
        alert_datetime_str = alert_datetime.isoformat()
        height, width, _ = alert.annotated_frame.shape
        alert_data = {
            'frame_id': alert.frame_id,
            'alert_msg': alert.alert_msg,
            'timestamp': alert.timestamp,
            'datetime': alert_datetime_str,
            'image': compressed_frame,
            'width': width,
            'height': height,
            'compression': 'jpeg',
        }

        # Log alert to file using the handle
        if self.log_file:
            self._log_alert(alert.frame_id, alert.alert_msg, alert.timestamp, alert_datetime_str)

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

    def _cleanup(self, alert_count: int):

        logger.info("NotificationsStreamWriter process terminating ...")
        logger.info(f"Total correctly processed alerts: {alert_count}")

        # Close log file
        if self.log_file:
            try:
                self.log_file.close()
                logger.info("Alert log file closed")
            except Exception as e:
                logger.error(f"Failed to close alert log file: {e}")

        # Close database (handles errors internally)
        if self.db_manager:
            self.db_manager.close()

        # Stop WebSocket server (handles errors internally)
        if self.ws_manager:
            self.ws_manager.stop()

    def run(self):
        """Main process loop."""

        alert_count = 0
        consecutive_failures = 0
        poison_pill_received = False

        ws_status = f"ws://{self.websocket_host}:{self.websocket_port}" if self.websocket_host else "disabled"
        db_status = self.database_url if self.database_url else "disabled"
        logfile_status = self.log_file_path if self.log_file_path else 'disabled'

        logger.info("NotificationsStreamWriter process starting")
        logger.info(f"Configuration:")
        logger.info(f"  - WebSocket: {ws_status}")
        logger.info(f"  - Database: {db_status}")
        logger.info(f"  - Log file: {logfile_status}")
        logger.info(f"  - JPEG quality: {self.alerts_jpeg_quality}\n")

        try:
            
            # Instantiate output managers
            # handles initialization errors internally
            self._setup_managers()

            # ---------------------------------
            # Alerts Processing
            # ---------------------------------

            while not self.error_event.is_set():

                try:
                    # Get alert from queue with timeout
                    alert = self.input_queue.get(timeout=self.alerts_get_timeout)
                except QueueEmptyException:
                    logger.debug("Input queue empty. Continuing to wait for alerts ... ")
                    continue

                try:
                    # if the alert being processed is the poison pill,
                    # break out of the loop to complete shutdown the process
                    if alert == POISON_PILL:
                        poison_pill_received = True
                        break

                    # Process the alert
                    self._process_alert(alert)
                    alert_count += 1
                    # reset counter
                    consecutive_failures = 0

                except Exception as e:
                    consecutive_failures += 1
                    if consecutive_failures < self.alerts_max_consecutive_failures:
                        logger.warning(
                            f"Error processing alert: {e}. "
                            f"Consecutive failures: {consecutive_failures} (max {self.alerts_max_consecutive_failures}). "
                            f"Will try to process next one ...", exc_info=True)
                    else:
                        self.error_event.set()
                        logger.error(
                            "Error event set: "
                            "threshold for max number of consecutive alerts failing to be processed has been surpassed. "
                            "Application will shutdown ..."
                        )
        
        except Exception as e:
            logger.critical(f"An unexpected critical error happened in notifications streamer process: {e}", exc_info=True)
            self.error_event.set()
            logger.warning("Error event set: force-stopping the application")

        finally:
            # final cleanup
            self._cleanup(alert_count)
            # log process conclusion
            logger.info(
                "NotificationsStreamWriter process stopped gracefully. "
                f"Poison pill received: {poison_pill_received}. "
                f"Error event: {self.error_event.is_set()}."
            )
            self.work_finished.set()


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
        alerts_jpeg_quality=85,
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

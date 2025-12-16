from typing import Optional
import logging
from sqlalchemy import create_engine, Column, Integer, String, Float, LargeBinary, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
import queue
import threading
from src.shared.processes.constants import (
    DB_MANAGER_THREAD_CLOSE_TIMEOUT,
    DB_MANAGER_QUEUE_WAIT_TIMEOUT,
    DB_POOL_SIZE,
    DB_MAX_OVERFLOW,
    DB_MANAGER_QUEUE_SIZE,
)


# ================================================================

logger = logging.getLogger("main.alert_out.db")

if not logger.handlers:  # Avoid duplicate handlers
    video_handler = logging.FileHandler('/app/logs/alert_out_db.log', mode='w')
    video_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(video_handler)
    logger.setLevel(logging.DEBUG)

# ================================================================


Base = declarative_base()


class Alert(Base):
    """SQLAlchemy model for storing alerts in the database."""
    __tablename__ = 'alerts'

    id = Column(Integer, primary_key=True, autoincrement=True)
    frame_id = Column(Integer, nullable=False, index=True)
    alert_msg = Column(String, nullable=False)
    timestamp = Column(Float, nullable=False, index=True)
    datetime = Column(DateTime, nullable=False, index=True)
    image_data = Column(LargeBinary, nullable=False)  # Compressed JPEG
    image_width = Column(Integer, nullable=False)
    image_height = Column(Integer, nullable=False)

    def __repr__(self):
        return f"<Alert(id={self.id}, frame_id={self.frame_id}, timestamp={self.timestamp})>"


class DatabaseManager:
    """Manages database operations for alert storage."""

    def __init__(
            self,
            database_url: Optional[str] = None,
            pool_size: int = DB_POOL_SIZE,
            max_overflow: int = DB_MAX_OVERFLOW,
            queue_get_timeout: float = DB_MANAGER_QUEUE_WAIT_TIMEOUT,
            thread_close_timeout: float = DB_MANAGER_THREAD_CLOSE_TIMEOUT,
            alerts_queue_size: int = DB_MANAGER_QUEUE_SIZE,
    ):
        """
        Initialize the database manager.

        Args:
            database_url: SQLAlchemy database URL
        """
        self.database_url = database_url        # always set, otherwise DB_manager not created
        self._db_engine = None
        self._db_session: Optional[Session] = None

        self.pool_size = pool_size
        self.max_overflow = max_overflow

        # Background worker components
        self._db_queue = queue.Queue(maxsize=alerts_queue_size)
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._queue_get_timeout = queue_get_timeout
        self._thread_close_timeout = thread_close_timeout

    def initialize(self):
        """Initialize database connection and create tables if they don't exist."""

        try:
            logger.info(f"Initializing database connection: {self.database_url}")
            self._db_engine = create_engine(
                self.database_url,
                pool_pre_ping=True,                     # Checks if connection is alive before using it
                pool_size=self.pool_size,               # Number of permanent connections
                max_overflow=self.max_overflow,         # Allow extra connections during spikes
                echo=False,
            )
            Base.metadata.create_all(self._db_engine)

            # Start the background worker thread
            self._stop_event.clear()
            self._worker_thread = threading.Thread(target=self._db_worker, daemon=True)
            self._worker_thread.start()
            logger.info("Database manager and worker thread initialized")

        except Exception as e:
            logger.error(f"Error initializing database: {e}", exc_info=True)

    def save_alert(self, **kwargs) -> bool:
        """
        Asynchronously queue an alert to be saved to the database.
        Returns True if queued, False if DB is disabled.
        """
        if self._db_engine is None:
            return False

        # Drop the data into the queue and return immediately
        try:
            self._db_queue.put_nowait(kwargs)
            logger.debug(
                "Alert saved in queue for database write: "
                f"Frame id: {kwargs.get('frame_id')}, "
                f"msg: {kwargs.get('alert_msg')}"
            )
            return True
        except queue.Full:
            logger.warning(
                f"Database queue is full (maxsize reached). "
                f"Dropping alert for Frame id: {kwargs.get('frame_id')}. "
                "Consider increasing DB_MANAGER_QUEUE_MAX_SIZE or optimizing DB performance."
            )
            return False

    def close(self):
        """Signal the worker to finish and close connections."""
        self._stop_event.set()

        if self._worker_thread:
            try:
                self._worker_thread.join(timeout=self._thread_close_timeout)
                logger.info("DB manager thread terminated successfully")
            except Exception as e:
                logger.error(f"Failed to terminate DB worker thread: {e}")

        if self._db_engine:
            try:
                self._db_engine.dispose()
                logger.info("Database engine disposed")
            except Exception as e:
                logger.error(f"Error disposing database engine: {e}")

    def _db_worker(self):
        """Background worker that handles the actual SQL I/O."""

        logger.info("Database background worker started")

        # Sessions are NOT thread-safe, so we create it inside the worker thread
        SessionFactory = sessionmaker(bind=self._db_engine)

        # continue to write alerts until the queue has not been emptied
        # even if the closing signal has been set
        # The thread may be killed if timeout is exceeded
        while not self._stop_event.is_set() or not self._db_queue.empty():

            # Wait for an alert with a timeout to check the stop_event
            try:
                alert_params = self._db_queue.get(timeout=self._queue_get_timeout)
            except queue.Empty:
                logger.debug("Alert queue empty. Continuing to wait for alert to appear ... ")
                continue

            # try to commit the alert to the DB
            # This outer try protects the thread from connection failures
            try:
                # 'with' clause handles __enter__ (connect), __exit__ (close), and automatic rollback on error.
                with SessionFactory() as session:
                    db_alert = Alert(**alert_params)
                    session.add(db_alert)
                    session.commit()
                    logger.info(
                        f"Committed a DB alert entry. "
                        f"Frame id: {alert_params['frame_id']}, "
                        f"msg: {alert_params['alert_msg']}"
                    )
            except (SQLAlchemyError, Exception) as e:
                # session.rollback() is handled automatically by the 'with' block if the error happened inside it.
                logger.error(f"DB Worker error for frame {alert_params.get('frame_id')}: {e}")
            finally:
                # Always signal task_done to allow the queue to drain properly
                self._db_queue.task_done()

        logger.info("Database background worker finished")

from typing import Optional
import logging
from sqlalchemy import create_engine, Column, Integer, String, Float, LargeBinary, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

# ================================================================

logger = logging.getLogger("main.alert_out")

if not logger.handlers:  # Avoid duplicate handlers
    video_handler = logging.FileHandler('/app/logs/alert_out.log', mode='w')
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

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize the database manager.

        Args:
            database_url: SQLAlchemy database URL
        """
        self.database_url = database_url
        self.db_session: Optional[Session] = None
        self.db_engine = None

    def initialize(self):
        """Initialize database connection and create tables if they don't exist."""
        if self.database_url is None:
            logger.info("Database URL not provided - database storage disabled")
            return

        try:
            logger.info(f"Initializing database connection: {self.database_url}")
            self.db_engine = create_engine(self.database_url, echo=False)
            Base.metadata.create_all(self.db_engine)
            SessionFactory = sessionmaker(bind=self.db_engine)
            self.db_session = SessionFactory()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}", exc_info=True)
            self.db_session = None
            self.db_engine = None

    def save_alert(
            self,
            frame_id: int,
            alert_msg: str,
            timestamp: float,
            datetime: str,
            image_data: bytes,
            image_width: int,
            image_height: int,
    ) -> bool:
        """
        Save an alert to the database.

        Args:
            frame_id: Frame identifier
            alert_msg: Alert message
            timestamp: Alert timestamp
            datetime: alert datetime
            image_data: Compressed JPEG image bytes
            image_width: Processed image width
            image_height: Processed image height

        Returns:
            True if successful, False otherwise
        """
        if self.db_session is None:
            return False

        try:
            db_alert = Alert(
                frame_id=frame_id,
                alert_msg=alert_msg,
                timestamp=timestamp,
                datetime=datetime,
                image_data=image_data,
                image_width=image_width,
                image_height=image_height,
            )

            self.db_session.add(db_alert)
            self.db_session.commit()
            logger.info(
                f"Alert saved to database: frame_id={frame_id}, "
                f"msg='{alert_msg}', size={len(image_data)} bytes"
            )
            return True

        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemy error saving alert: {e}", exc_info=True)
            self.db_session.rollback()
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving to database: {e}", exc_info=True)
            self.db_session.rollback()
            return False

    def close(self):
        """Close database connections."""
        if self.db_session:
            try:
                self.db_session.close()
                logger.info("Database session closed")
            except Exception as e:
                logger.error(f"Error closing database session: {e}")

        if self.db_engine:
            try:
                self.db_engine.dispose()
                logger.info("Database engine disposed")
            except Exception as e:
                logger.error(f"Error disposing database engine: {e}")


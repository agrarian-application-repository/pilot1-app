import os
import time
import logging
import multiprocessing as mp
from src.shared.processes.constants import (
    VIDEO_OUT_STORE_MAX_UPLOAD_RETRIES,
    VIDEO_OUT_STORE_RETRY_BACKOFF_TIME,
    POISON_PILL,
)

# ================================================================
logger = logging.getLogger("main.video_out.storage")

if not logger.handlers:  # Avoid duplicate handlers
    video_handler = logging.FileHandler('./logs/video_out_storage.log', mode='w')
    video_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(video_handler)
    logger.setLevel(logging.DEBUG)


# ================================================================

class VideoPersistenceProcess(mp.Process):
    """
    Dedicated process that waits for video file paths from a queue and
    uploads them to remote storage.
    """

    def __init__(
            self,
            input_queue: mp.Queue,
            storage_url: str,
            delete_local_on_success: bool = True,
            max_retries: int = VIDEO_OUT_STORE_MAX_UPLOAD_RETRIES,
            retry_backoff: float = VIDEO_OUT_STORE_RETRY_BACKOFF_TIME,
    ):
        super().__init__()
        self.input_queue = input_queue
        self.storage_url = storage_url
        self.delete_local_on_success = delete_local_on_success

        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

        self.work_finished = mp.Event()
        
    def _cleanup_local_file(self, video_file_path: str):
        """Safely removes the local file on upload success, if configured to do so."""
        if self.delete_local_on_success:
            try:
                if os.path.exists(video_file_path):
                    os.remove(video_file_path)
                    logger.info(f"Successfully deleted local file: {video_file_path}")
            except OSError as e:
                logger.error(f"Could not delete local file {video_file_path}: {e}")

    def _upload_file(self, file: str, url: str) -> bool:
        """
        Abstract method for service-specific storage logic.
        Should return True on success, False on failure.
        """
        # This depends on the service used to store the video (Azure, S3, etc.)
        raise NotImplementedError
        # TODO
    
    def _run_upload_routine(self, video_file_path: str):
        """
        Try to perform the upload with constant backoff.
        """
        
        if not os.path.exists(video_file_path):
            logger.error(f"Upload aborted: Local file {video_file_path} not found.")
            return

        attempt = 0
        success = False

        while attempt < self.max_retries:

            try:
                logger.info(f"Starting upload attempt {attempt + 1}/{self.max_retries}...")
                success = self._upload_file(file=video_file_path, url=self.storage_url)

                if success:
                    logger.info(f"Remote upload complete: {self.storage_url}")
                    self._cleanup_local_file(video_file_path)
                    break
                else:
                    logger.warning(f"Storage service returned failure for {video_file_path}")

            except Exception as e:
                logger.error(f"Unexpected error during upload attempt {attempt + 1}: {e}")

            attempt += 1
            if attempt < self.max_retries:
                logger.info(f"Retrying upload in {self.retry_backoff} seconds...")
                time.sleep(self.retry_backoff)

        if not success:
            logger.error(
                f"Failed to persist {video_file_path} after {self.max_retries} attempts. "
                "Local file is preserved for manual recovery."
            )

    def run(self):
        """
        Process entry point. 
        Loops until a poison pill is received.
        """
        logger.info("VideoPersistenceProcess started and waiting for tasks...")

        while True:

            try:

                # Block until a message arrives
                message = self.input_queue.get()

                if message == POISON_PILL:
                    logger.info("Poison pill received. Shutting down Persistence Process.")
                    break

                # If message is not a pill, it's the locally saved video path
                logger.info(f"Received upload task for: {message}")
                self._run_upload_routine(message)

            except Exception as e:
                logger.error(
                    f"Critical error in Persistence Process loop: {e}. "
                    "Continuing to wait for the next video to upload or the poison pill."
                )
                continue

        self.work_finished.set()








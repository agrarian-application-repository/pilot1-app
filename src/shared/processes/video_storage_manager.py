import os
import time
import logging
import shutil
import multiprocessing as mp
from src.shared.processes.constants import *
from queue import Empty as QueueEmptyException

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
            delete_local_on_success: bool = VIDEO_OUT_STORE_DELETE_LOCAL_ON_SUCCESS,
            queue_get_timeout: float = VIDEO_OUT_STORE_QUEUE_GET_TIMEOUT,
            max_retries: int = VIDEO_OUT_STORE_MAX_UPLOAD_RETRIES,
            retry_backoff: float = VIDEO_OUT_STORE_RETRY_BACKOFF_TIME,
    ):
        super().__init__()
        self.input_queue = input_queue
        self.storage_url = storage_url
        self.delete_local_on_success = delete_local_on_success
        self.queue_get_timeout = queue_get_timeout
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
        raise NotImplementedError("No service-specific upload logic implemented.")
    
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

        try:

            while True:

                try:
                    message = self.input_queue.get(timeout=self.queue_get_timeout)
                except QueueEmptyException:
                    logger.debug("No upload tasks in queue, continuing to wait...")
                    continue

                if message == POISON_PILL:
                    logger.info("Poison pill received. Shutting down Persistence Process.")
                    break

                # If message is not a pill, it's the locally saved video path
                logger.info(f"Received upload task for: {message}")
                self._run_upload_routine(message)

        except Exception as e:
            logger.error(
                f"Critical error in Persistence Process loop: {e}. Terminating process.",
                exc_info=True,
            )
        
        finally:
            logger.info("VideoPersistenceProcess terminated gracefully.")
            self.work_finished.set()



class LocalCopyVideoPersistenceProcess(VideoPersistenceProcess):
    """
    VideoPersistenceProcess implementation that saves files to a local directory.
    """

    def _upload_file(self, file: str, url: str) -> bool:
        """
        Local copy implementation of the upload routine.
        Simply moves the file to the target directory.
        """

        time.sleep(5)  # Simulate some delay

        try:
            target_path = os.path.join(url, os.path.basename(file))
            shutil.copy2(file, target_path)
            logger.info(f"File copied to local storage: {target_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to copy file {file} to {url}: {e}")
            return False


if __name__ == "__main__":

    import multiprocessing as mp
    import time

    # Example usage of LocalCopyVideoPersistenceProcess
    input_queue = mp.Queue(maxsize=MAX_SIZE_VIDEO_STORAGE)
    storage_directory = "./local_video_storage"

    # Ensure the storage directory exists
    os.makedirs(storage_directory, exist_ok=True)

    video_persistence_process = LocalCopyVideoPersistenceProcess(
        input_queue=input_queue,
        storage_url=storage_directory,
    )

    video_persistence_process.start()

    # Simulate adding video files to the queue
    for i in range(5):
        fake_video_path = f"./video_{i}.mp4"
        # Create a dummy file to simulate a video file
        with open(fake_video_path, 'w') as f:
            f.write("This is a dummy video file.\n")
        input_queue.put(fake_video_path, timeout=1.0)
        print(f"Enqueued {fake_video_path} for upload.")
        time.sleep(1)


    # Send poison pill to terminate the process
    time.sleep(2)
    input_queue.put(POISON_PILL)

    # Wait for the process to finish
    video_persistence_process.join()
    print("Video persistence process has terminated.")

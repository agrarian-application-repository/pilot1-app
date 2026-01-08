import threading
import json
import time
import logging
from queue import Queue
import websocket        # pip install websocket-client
import base64
import cv2
import numpy as np


# ================================================================

logger = logging.getLogger("ui.receiver")
log_path = "./logs/ui_receiver.log"

if not logger.handlers:  # Avoid duplicate handlers
    alert_handler = logging.FileHandler(log_path, mode='w')
    alert_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(alert_handler)
    logger.setLevel(logging.DEBUG)

# ================================================================


class AlertReceiver(threading.Thread):

    def __init__(
            self,
            host: str,
            port: int,
            shared_queue: Queue,
            reconnection_delay: int,
            ping_interval: int,
            ping_timeout: int
    ):

        super().__init__(daemon=True)  # Thread dies when main process exits
        self.uri = f"ws://{host}:{port}"
        self.shared_queue = shared_queue
        self.stop_signal = threading.Event()
        self.ws = None

        self.reconnection_delay = reconnection_delay
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            if 'image' in data and data['image']:
                data['image'] = self.decompress_image(data['image'])
            self.shared_queue.put(data)
        except Exception as e:
            logger.error(f"Error parsing alert message: {e}")

    def on_error(self, ws, error):
        logger.error(f"WebSocket Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        logger.warning(f"Connection closed: {close_status_code} - {close_msg}")

    def decompress_image(self, b64_string: str):
        """
        Converts a base64 encoded JPEG string into a BGR NumPy array.
        """
        try:
            # 1. Decode base64 to bytes
            img_bytes = base64.b64decode(b64_string)
            # 2. Convert bytes to a 1D numpy array
            nparr = np.frombuffer(img_bytes, np.uint8)
            # 3. Decode JPEG to image (BGR format)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img_bgr is None:
                logger.error("Failed to decompress image: imdecode returned None")
                return None

            # 4. Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return img_rgb

        except Exception as e:
            logger.error(f"Image decompression error: {e}")
            return None

    def run(self):
        """Main loop with auto-reconnect logic."""
        while not self.stop_signal.is_set():
            try:
                logger.info(f"Connecting to {self.uri}...")
                self.ws = websocket.WebSocketApp(
                    self.uri,
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close
                )
                self.ws.run_forever(ping_interval=self.ping_interval, ping_timeout=self.ping_timeout)
            except Exception as e:
                logger.error(f"Connection failed: {e}")

            # Wait before attempting to reconnect
            if not self.stop_signal.is_set():
                time.sleep(self.reconnection_delay)

    def stop(self):
        self.stop_signal.set()
        if self.ws:
            self.ws.close()

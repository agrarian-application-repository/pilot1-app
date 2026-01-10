import threading
import json
import time
import logging
import websocket        # pip install websocket-client
import base64
import cv2
import numpy as np
import asyncio
import websockets
from collections import deque
from matplotlib import pyplot as plt

# ================================================================

logger = logging.getLogger("ui.receiver")
log_path = "./logs/ui/ui_receiver.log"

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
            shared_dequeue: deque,
            reconnection_delay: int,
            ping_interval: int,
            ping_timeout: int
    ):

        super().__init__(daemon=True)  # Thread dies when main process exits
        self.uri = f"ws://{host}:{port}"
        self.shared_dequeue = shared_dequeue
        self.stop_signal = threading.Event()
        self.ws = None
        self.total_alerts = 0

        self.reconnection_delay = reconnection_delay
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout

    def get_total_alerts(self):
        return self.total_alerts

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            if 'image' in data and data['image']:
                data['image'] = self.decompress_image(data['image'])
            self.shared_dequeue.appendleft(data)
            self.total_alerts += 1
        except Exception as e:
            logger.error(f"Error parsing alert message: {e}")

    def on_error(self, ws, error):
        logger.error(f"WebSocket Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        logger.warning(f"Connection closed: {close_status_code} - {close_msg}")

    def decompress_image(self, b64_string: str):
        """
        Converts a base64 encoded JPEG string into a RGB NumPy array.
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


async def mock_server(host, port):
    """A simple websocket server that sends a dummy image every 2 seconds."""

    async def handler(websocket):
        print("Mock Server: Client connected.")

        try:
            for i in range(30):  # Send 30 messages then stop

                # Create a dummy RGB image (blue square)
                dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
                random_color = np.random.randint(0, 256, size=3, dtype=np.uint8)
                dummy_img[:, :] = random_color
                _, buffer = cv2.imencode('.jpg', dummy_img)
                b64_image = base64.b64encode(buffer).decode('utf-8')

                payload = {
                    "timestamp": time.time(),
                    "image": b64_image,
                    "metadata": f"Alert #{i}"
                }
                await websocket.send(json.dumps(payload))
                print(f"Mock Server: Sent Alert #{i}")
                await asyncio.sleep(2)
        except websockets.ConnectionClosed:
            pass
        print("Mock Server: Finished sending data.")

    async with websockets.serve(handler, host, port):
        await asyncio.Future()  # Run forever


def run_server():
    asyncio.run(mock_server("localhost", 8765))


def save_images_left_to_right(images_deque, output_path, dpi=100):

    images = [elem.get('image') for elem in images_deque]
    images = [img for img in images if img is not None]

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3), dpi=dpi)

    # Handle single image case
    if n == 1:
        axes = [axes]

    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout(pad=0)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    # 1. Start Mock Server in a background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(1)  # Give server a moment to start

    # 2. Initialize the AlertReceiver components
    shared_dequeue = deque(maxlen=5)
    receiver = AlertReceiver(
        host="localhost",
        port=8765,
        shared_dequeue=shared_dequeue,
        reconnection_delay=5,
        ping_interval=10,
        ping_timeout=5,
    )

    # 3. Start the Receiver thread
    print("Main: Starting AlertReceiver...")
    receiver.start()

    # 4. Consume data from the queue
    try:
        alerts_processed = 0
        while alerts_processed < 30:
            if receiver.get_total_alerts() > alerts_processed:
                print(f"Main: Received image")
                save_images_left_to_right(shared_dequeue, f"alert_{alerts_processed}.png")
                # Optional: Show the image to verify decompression
                # cv2.imshow("Received Alert", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                # cv2.waitKey(1)

                alerts_processed += 1
            else:
                time.sleep(0.1)  # Small sleep to prevent CPU spiking

    except KeyboardInterrupt:
        print("Main: Stopping test...")
    finally:
        # 5. Cleanup
        receiver.stop()
        receiver.join(timeout=2)
        cv2.destroyAllWindows()
        print("Main: Test complete.")

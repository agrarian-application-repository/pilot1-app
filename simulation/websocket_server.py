import threading
import json
import time
import base64
import cv2
import numpy as np
import asyncio
import websockets
from datetime import datetime
import random


async def mock_server(host, port):
    """A simple websocket server that sends a dummy image every 2 seconds."""

    async def handler(websocket):
        print("Mock Server: Client connected.")

        # black frame
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # 3. Define text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 100)  # Coordinates (X, Y) where text starts
        fontScale = 2
        color = (255, 255, 255)  # White in BGR
        thickness = 3

        try:
            for i in range(30):  # Send 30 messages then stop

                dummy_img = frame.copy()
                # put current datetime string oin frame
                timestamp = time.time()
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(dummy_img, current_time, org, font, fontScale, color, thickness, cv2.LINE_AA)

                _, buffer = cv2.imencode('.jpg', dummy_img)
                b64_image = base64.b64encode(buffer).decode('utf-8')

                payload = {
                    'frame_id': i,
                    'alert_msg': random.choice(["road", "car", "slope"]),
                    'timestamp': timestamp,
                    'datetime': current_time,
                    'image': b64_image,
                    'width': 1920,
                    'height': 1080,
                    'compression': 'jpeg',
                }

                await websocket.send(json.dumps(payload))
                print(f"Mock Server: Sent Alert #{i}")
                await asyncio.sleep(5)
        except websockets.ConnectionClosed:
            pass
        print("Mock Server: Finished sending data.")

    async with websockets.serve(handler, host, port):
        await asyncio.Future()  # Run forever


def run_server():
    asyncio.run(mock_server("localhost", 8765))


if __name__ == "__main__":
    # 1. Start Mock Server in a background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(1)  # Give server a moment to start

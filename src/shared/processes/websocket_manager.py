import multiprocessing as mp
import asyncio
import json
from collections import deque
from typing import Optional, Set
import websockets
from websockets.server import serve
import threading
import queue
import logging


# ================================================================

logger = logging.getLogger("main.alert_out")

if not logger.handlers:  # Avoid duplicate handlers
    video_handler = logging.FileHandler('/app/logs/alert_out.log', mode='w')
    video_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(video_handler)
    logger.setLevel(logging.DEBUG)

# ================================================================


class WebSocketManager:
    """Manages WebSocket server and client connections."""

    def __init__(
            self,
            host: str = "localhost",
            port: int = 8765,
            stop_event: Optional[mp.Event] = None
    ):
        """
        Initialize the WebSocket manager.

        Args:
            host: Host address for the server
            port: Port number for the server
            stop_event: Event to signal server shutdown
        """
        self.host = host
        self.port = port
        self.stop_event = stop_event
        self.connected_clients: Set = set()
        self.alert_queue = queue.Queue()
        self.recent_alerts = deque()
        self._server_thread = None

    def set_recent_alerts(self, recent_alerts: deque):
        """
        Set the deque for storing recent alerts.

        Args:
            recent_alerts: Deque to store recent alerts
        """
        self.recent_alerts = recent_alerts

    def queue_alert(self, alert_data: dict):
        """
        Queue an alert for broadcasting.

        Args:
            alert_data: Alert data dictionary
        """
        self.alert_queue.put(alert_data)
        logger.debug(f"Alert queued for broadcast: frame_id={alert_data['frame_id']}")

    async def _handle_client(self, websocket):
        """
        Handle a WebSocket client connection.

        Args:
            websocket: WebSocket connection
        """
        self.connected_clients.add(websocket)
        client_addr = websocket.remote_address
        logger.info(f"Client connected from {client_addr}. Total clients: {len(self.connected_clients)}")

        try:
            # Send recent alerts to new client
            recent_count = len(self.recent_alerts)
            if recent_count > 0:
                logger.info(f"Sending {recent_count} recent alerts to new client {client_addr}")
                for alert in self.recent_alerts:
                    await websocket.send(json.dumps(alert))

            # Keep connection alive and handle incoming messages
            async for message in websocket:
                logger.debug(f"Received message from {client_addr}: {message[:100]}")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_addr} connection closed")
        except Exception as e:
            logger.error(f"Error handling client {client_addr}: {e}", exc_info=True)
        finally:
            self.connected_clients.remove(websocket)
            logger.info(f"Client {client_addr} disconnected. Total clients: {len(self.connected_clients)}")

    async def _broadcast_alerts(self):
        """Continuously broadcast new alerts to all connected clients."""
        logger.info("Alert broadcast task started")

        while not self.stop_event.is_set():
            try:
                # Check for new alerts (non-blocking with timeout)
                try:
                    alert_data = self.alert_queue.get(timeout=0.1)
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue

                # Broadcast to all connected clients
                if self.connected_clients:
                    message = json.dumps(alert_data)
                    client_count = len(self.connected_clients)
                    logger.info(f"Broadcasting alert (frame_id={alert_data['frame_id']}) "
                                     f"to {client_count} client(s)")

                    disconnected = set()

                    for client in self.connected_clients:
                        try:
                            await client.send(message)
                        except websockets.exceptions.ConnectionClosed:
                            disconnected.add(client)
                            logger.debug(f"Client {client.remote_address} disconnected during broadcast")

                    # Remove disconnected clients
                    self.connected_clients -= disconnected
                    if disconnected:
                        logger.info(f"Removed {len(disconnected)} disconnected client(s)")
                else:
                    logger.debug(f"No clients connected - alert not broadcast (frame_id={alert_data['frame_id']})")

            except Exception as e:
                logger.error(f"Error broadcasting alert: {e}", exc_info=True)
                await asyncio.sleep(0.1)

        logger.info("Alert broadcast task stopped")

    async def _run_server(self):
        """Run the WebSocket server."""
        logger.info(f"Starting WebSocket server on ws://{self.host}:{self.port}")

        async with serve(self._handle_client, self.host, self.port):
            logger.info(f"WebSocket server listening on ws://{self.host}:{self.port}")

            # Run broadcast task
            broadcast_task = asyncio.create_task(self._broadcast_alerts())

            # Wait for stop event
            while not self.stop_event.is_set():
                await asyncio.sleep(0.1)

            logger.info("Stop event received - shutting down WebSocket server")

            # Cleanup
            broadcast_task.cancel()
            try:
                await broadcast_task
            except asyncio.CancelledError:
                logger.debug("Broadcast task cancelled successfully")

    def _run_async_loop(self):
        """Run the asyncio event loop in a separate thread."""
        logger.debug("Starting asyncio event loop in separate thread")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self._run_server())
        except Exception as e:
            logger.error(f"Error in async loop: {e}", exc_info=True)
        finally:
            loop.close()
            logger.debug("Asyncio event loop closed")

    def start(self):

        """
        If WebSocket isn used, Start the WebSocket server in a separate thread.
        """

        self._server_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._server_thread.start()
        logger.info("WebSocket server thread started")

    def stop(self):
        """Stop the WebSocket server and close all connections."""

        client_count = len(self.connected_clients)
        if client_count > 0:
            logger.info(f"Closing {client_count} WebSocket connection(s)")
            for client in list(self.connected_clients):
                try:
                    asyncio.run(client.close())
                except Exception as e:
                    logger.debug(f"Error closing client connection: {e}")

        # Wait for server thread to finish
        if self._server_thread:
            logger.info("Waiting for WebSocket thread to terminate...")
            self._server_thread.join(timeout=2)
            if self._server_thread.is_alive():
                logger.warning("WebSocket thread did not terminate cleanly")
            else:
                logger.info("WebSocket thread terminated successfully")


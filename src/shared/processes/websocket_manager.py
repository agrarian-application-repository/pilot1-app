import multiprocessing as mp
import asyncio
import json
from collections import deque
from typing import Set
import websockets
from websockets.server import serve
import threading
import queue
import logging
from src.shared.processes.constants import (
    WS_COMMON_PORT,
    WS_PORT,
    WSS_PORT,
    WS_MANAGER_THREAD_CLOSE_TIMEOUT,
    WS_MANAGER_QUEUE_WAIT_TIMEOUT,
    WS_MANAGER_STOP_EVENT_CHECK_PERIOD,
)


# ================================================================

logger = logging.getLogger("main.alert_out.ws")

if not logger.handlers:  # Avoid duplicate handlers
    video_handler = logging.FileHandler('/app/logs/alert_out_ws.log', mode='w')
    video_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(video_handler)
    logger.setLevel(logging.DEBUG)

# ================================================================


class WebSocketManager:
    """Manages WebSocket server and client connections."""

    def __init__(
            self,
            error_event: mp.Event,
            host: str = "localhost",
            port: int = WSS_PORT,
            queue_wait_timeout: float = WS_MANAGER_QUEUE_WAIT_TIMEOUT,
            stop_event_check_period: float = WS_MANAGER_STOP_EVENT_CHECK_PERIOD,
            thread_close_timeout: float = WS_MANAGER_THREAD_CLOSE_TIMEOUT,

    ):
        """
        Initialize the WebSocket manager.

        Args:
            host: Host address for the server
            port: Port number for the server
            error_event: Event to signal server shutdown
        """
        self.host = host
        self.port = port
        self.error_event = error_event
        self.connected_clients: Set = set()
        self.alert_queue = queue.Queue()
        self.recent_alerts = deque()
        self.recent_alerts_lock = None

        self.queue_wait_timeout = queue_wait_timeout
        self.stop_event_check_period = stop_event_check_period
        self.thread_close_timeout = thread_close_timeout

        # Threading and Asyncio control
        self._server_thread = None
        self._loop = None
        self._stop_event_async = None   # Internal asyncio-native event

    def set_recent_alerts(self, recent_alerts: deque, lock: threading.Lock):
        """
        Set the deque where recent alerts are stored.

        Args:
            recent_alerts: Deque where recent alerts are stored
            lock: A threading lock to prevent simultaneous write/read operations by the main process and the WS manager
        """
        self.recent_alerts = recent_alerts
        self.recent_alerts_lock = lock

    def queue_alert(self, alert_data: dict):
        """
        Queue an alert for broadcasting.

        Args:
            alert_data: Alert data dictionary
        """
        self.alert_queue.put(alert_data)
        logger.debug(f"Alert queued for broadcast: frame_id={alert_data['frame_id']}")

    def _is_stop_requested(self) -> bool:
        """Checks both the local async event (stop by alert writer process) and the global error_event (mp.Event)."""
        internal_set = self._stop_event_async.is_set() if self._stop_event_async else False
        return internal_set or self.error_event.is_set()

    async def _handle_client(self, websocket):
        """
        Handle a WebSocket client connection.

        Args:
            websocket: WebSocket connection
        """
        self.connected_clients.add(websocket)
        client_addr = websocket.remote_address
        logger.info(
            f"Client connected from {client_addr}. "
            f"Total clients: {len(self.connected_clients)}"
        )

        try:

            # Thread-safe snapshotting of the recent alerts
            with self.recent_alerts_lock:
                recent_alerts_snapshot = list(self.recent_alerts)
                recent_count = len(recent_alerts_snapshot)

            # Send recent alerts to new client
            if recent_count > 0:
                logger.info(f"Sending {recent_count} recent alerts")
                for alert in recent_alerts_snapshot:
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
            logger.info(
                f"Client {client_addr} disconnected. "
                f"Total clients: {len(self.connected_clients)}"
            )

    async def _broadcast_alerts(self):
        """Continuously broadcast new alerts to all connected clients."""
        logger.info("Alert broadcast task started")

        while not self._is_stop_requested():

            try:
                # 1. Use a non-blocking check first
                if self.alert_queue.empty():
                    # This sleep gives the event loop time to breathe/handle clients
                    await asyncio.sleep(self.queue_wait_timeout)
                    continue

                # 2. If not empty, get the data (now it won't block)
                alert_data = self.alert_queue.get_nowait()

                # Broadcast to all connected clients
                if self.connected_clients:
                    message = json.dumps(alert_data)
                    client_count = len(self.connected_clients)
                    logger.info(f"Broadcasting alert (frame_id={alert_data['frame_id']}) to {client_count} client(s)")

                    # Parallel send to all clients
                    tasks = [client.send(message) for client in list(self.connected_clients)]
                    await asyncio.gather(*tasks, return_exceptions=True)
                    # collect disconnection exceptions

                    # no need to explicitely remove clients manually because the _handle_client() task
                    # is already doing it "reactively" in the background.

                else:
                    logger.debug(f"No clients connected - alert not broadcast (frame_id={alert_data['frame_id']})")

            except Exception as e:
                logger.error(f"Error broadcasting alert: {e}", exc_info=True)
                await asyncio.sleep(0.1)

        logger.info("Alert broadcast task stopped")

    async def _run_server(self):
        """Run the WebSocket server."""
        logger.info(f"Starting WebSocket server on ws://{self.host}:{self.port}")

        # This event MUST be created inside the event loop
        self._stop_event_async = asyncio.Event()

        async with serve(self._handle_client, self.host, self.port):
            logger.info(f"WebSocket server listening on ws://{self.host}:{self.port}")

            # Run broadcast task
            broadcast_task = asyncio.create_task(self._broadcast_alerts())

            # Loop until either the local stop event or global error event triggers
            while not self._is_stop_requested():
                await asyncio.sleep(self.stop_event_check_period)

            logger.info("Shutdown signal detected - cleaning up WebSocket server")
            broadcast_task.cancel()
            try:
                await broadcast_task
            except asyncio.CancelledError:
                logger.debug("Broadcast task cancelled successfully")
                pass

    def _run_async_loop(self):
        """Run the asyncio event loop in a separate thread."""

        logger.debug("Starting asyncio event loop in separate thread")
        
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            self._loop.run_until_complete(self._run_server())
        except Exception as e:
            logger.error(f"Error in async loop: {e}", exc_info=True)
        finally:
            self._loop.close()
            logger.debug("Asyncio event loop closed")

    def start(self):

        """
        If WebSocket isn used, Start the WebSocket server in a separate thread.
        """

        self._server_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._server_thread.start()
        logger.info("WebSocket server thread started")

    def stop(self):
        """Cleanly disconnects clients and shuts down the thread."""

        # 1. Signal the async loop thread-safely
        if self._loop and self._stop_event_async:
            self._loop.call_soon_threadsafe(self._stop_event_async.set)
            logger.info("Stop event async set for termination")

        # 2. Join the thread to ensure it finished cleanup
        if self._server_thread:
            logger.info("Waiting for WebSocket thread to terminate...")
            self._server_thread.join(timeout=self.thread_close_timeout)
            if self._server_thread.is_alive():
                logger.warning("WebSocket thread did not terminate cleanly")
            else:
                logger.info("WebSocket thread terminated successfully")


import asyncio
import json
import logging
import threading
from typing import Optional, Dict, Set
import websockets
from websockets.server import serve

from src.shared.processes.constants import *

# ================================================================

logger = logging.getLogger("main.alert_out.ws")

if not logger.handlers:  # Avoid duplicate handlers
    video_handler = logging.FileHandler('/app/logs/alert_out_ws.log', mode='w')
    video_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(video_handler)
    logger.setLevel(logging.DEBUG)


# ===============================================================)


class WebSocketManager:
    """
    Manages WebSocket server and client connections for real-time alert broadcasting.

    This manager is designed to run in a background thread of an alert-generating process.
    It optimizes bandwidth by only storing and syncing the 'last alert' generated.
    It includes a heartbeat mechanism to detect and prune dead client connections.
    """

    def __init__(
            self,
            host: str = "localhost",
            port: int = WSS_PORT,
            ping_interval: float = WS_MANAGER_PING_INTERVAL,
            ping_timeout: float = WS_MANAGER_PING_TIMEOUT,
            broadcast_timeout: float = WS_MANAGER_BROADCAST_TIMEOUT,
            thread_close_timeout: float = WS_MANAGER_THREAD_CLOSE_TIMEOUT,
    ):
        """
        Initialize the WebSocket manager.

        Args:
            host: Host address for the server.
            port: Port number for the server.
            ping_interval: Seconds between heartbeat pings.
            ping_timeout: Seconds to wait for a pong before closing connection.
            broadcast_timeout: timeout for canceling the sending of a message to a single client
            thread_close_timeout: Seconds to wait for thread join during stop().
        """
        self.host = host
        self.port = port

        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.connected_clients: Set = set()

        self.broadcast_timeout = broadcast_timeout
        self.thread_close_timeout = thread_close_timeout

        # Shared state protected by internal lock
        self._last_alert: Optional[Dict] = None
        self._lock = threading.Lock()

        # Threading and Asyncio control
        self._server_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._new_alert_event: Optional[asyncio.Event] = None

    def queue_alert(self, alert_data: Dict):
        """
        Thread-safe method to update the latest alert and trigger a broadcast.
        Can be called directly by the main alert-generation process.

        Args:
            alert_data: Alert data dictionary.
        """
        with self._lock:
            self._last_alert = alert_data

        # Signal the async event loop thread-safely (if _new_alert_event exists, so does _stop_event)
        if self._loop and self._new_alert_event and not self._stop_event.is_set():
            self._loop.call_soon_threadsafe(self._new_alert_event.set)
            logger.debug(f"Alert queued for broadcast: frame_id={alert_data.get('frame_id')}")

    async def _handle_client(self, websocket):
        """
        Handle a WebSocket client connection lifecycle.
        Sends the most recent alert immediately upon connection (sync).
        """
        client_addr = websocket.remote_address
        self.connected_clients.add(websocket)
        logger.info(
            f"Client connected: {client_addr}. "
            f"Total clients: {len(self.connected_clients)}"
        )

        try:
            # 1. Immediate State Sync: Send the latest alert snapshot
            with self._lock:
                snapshot = self._last_alert

            # avoid sending alert if none exist yet
            if snapshot:
                await websocket.send(json.dumps(snapshot))
                logger.debug(f"Synced last alert to new client {client_addr}")

            # 2. Keep connection alive.
            # The 'websockets' library handles pings/pongs automatically in the background.
            async for message in websocket:
                logger.debug(f"Received unexpected message from {client_addr}: {message[:50]}")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed by {client_addr}")
        except Exception as e:
            logger.error(f"Error handling client {client_addr}: {e}", exc_info=True)
        finally:
            self.connected_clients.discard(websocket)
            logger.info(f"Client {client_addr} disconnected. Total: {len(self.connected_clients)}")

    async def _broadcast_loop(self):
        """Continuously broadcast the latest alert to all connected clients."""
        logger.info("Broadcast loop started")

        while not self._stop_event.is_set():

            # Wait for the signal from queue_alert() or stop()
            # alert event can be set in stop() to ensure this await unblocks
            # and break terminates the loop immediately
            await self._new_alert_event.wait()
            self._new_alert_event.clear()  # reset signal to False

            # terminate broadcasting when the alert event is faked by stop()
            if self._stop_event.is_set():
                break

            # acquire lock to get the alert reference
            with self._lock:
                # if there are no alerts or no clients yet, just return to wait
                if not self._last_alert or not self.connected_clients:
                    continue
                # Get the reference (atomic/fast)
                snapshot = self._last_alert

            # unlock and do the heavy serialization
            # avoids main process having to wait to put a new alert
            try:
                message = json.dumps(snapshot)
                frame_id = snapshot.get('frame_id')
            except Exception as e:
                logger.error(f"Failed to serialize alert: {e}")
                continue

            # Broadcast in parallel to all clients
            logger.info(f"Broadcasting alert {frame_id} to {len(self.connected_clients)} client(s)")
            tasks = [asyncio.create_task(client.send(message)) for client in self.connected_clients]

            if tasks:
                # Use a timeout to prevent slow clients from blocking the loop
                done, pending = await asyncio.wait(tasks, timeout=self.broadcast_timeout)
                for task in pending:
                    task.cancel()
                logger.warning(
                    f"Cancelled {len(pending)} out of {len(done)+len(pending)} broadcasting tasks, "
                    f"timeout exceeded"
                )

        logger.info("Broadcast loop stopped")

    async def _run_server(self):
        """Main entry point for the WebSocket server inside the event loop."""
        self._stop_event = asyncio.Event()
        self._new_alert_event = asyncio.Event()

        # serve() includes built-in heartbeat/ping functionality
        async with serve(
                self._handle_client,
                self.host,
                self.port,
                ping_interval=self.ping_interval,
                ping_timeout=self.ping_timeout,
        ):
            logger.info(f"WebSocket server active on ws://{self.host}:{self.port}")

            broadcast_task = asyncio.create_task(self._broadcast_loop())

            # Wait for shutdown signal
            await self._stop_event.wait()

            logger.info("Shutdown signal detected - cleaning up WebSocket server")
            broadcast_task.cancel()
            try:
                await broadcast_task
                logger.info("Successfully cancelled broadcasting task")
            except asyncio.CancelledError:
                logger.error("Failed to cancel broadcasting task")

    def _run_async_loop(self):
        """Run the asyncio event loop in the separate thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._run_server())
        except Exception as e:
            logger.error(f"Error in WebSocket event loop: {e}", exc_info=True)
        finally:
            self._loop.close()
            logger.debug("Asyncio event loop closed")

    def start(self):
        """Start the WebSocket server in a separate thread."""
        self._server_thread = threading.Thread(
            target=self._run_async_loop,
            name="WS_Manager_Thread",
            daemon=True
        )
        self._server_thread.start()
        logger.info("WebSocket server thread started")

    def stop(self):
        """
        Cleanly disconnects clients and shuts down the thread.
        """

        if self._loop and self._stop_event:
            # Signal server and broadcaster to stop
            self._loop.call_soon_threadsafe(self._stop_event.set)
            self._loop.call_soon_threadsafe(self._new_alert_event.set)
            logger.info("Stop signal sent to WebSocket event loop")

        if self._server_thread:
            logger.info("Waiting for WebSocket thread to terminate...")
            self._server_thread.join(timeout=self.thread_close_timeout)
            if self._server_thread.is_alive():
                logger.warning("WebSocket thread did not terminate cleanly within timeout")
            else:
                logger.info("WebSocket thread terminated successfully")

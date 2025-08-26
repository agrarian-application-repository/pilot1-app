import socket
import json
import time
import logging
import multiprocessing as mp
from src.shared.processes.messages import TelemetryQueueObject

# ================================================================

logger = logging.getLogger("main.stream_telemetry_in")

if not logger.handlers:  # Avoid duplicate handlers
    video_handler = logging.FileHandler('/app/logs/stream_telemetry_in.log')
    video_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(video_handler)
    logger.setLevel(logging.DEBUG)

# ================================================================

class StreamTelemetryListener(mp.Process):
    """
    A Process that listens for telemetry messages (JSON) over UDP.
    Messages are stored them with a timestamp in a queue.
    TelemetyListener stop event will be set by the StreamVideoReader process terminates
    """
    
    def __init__(
            self, 
            port: int, 
            telemetry_queue: mp.Queue,
    ):
        """
        Initialize the StreamTelemetryListener process.
        
        Args:
            port (int): port where UDP packets are received"
            telemetry_queue (mp.Queue): The mp.Queue where to append received packets
        """
        super().__init__()
        
        self._stop_event = mp.Event()
        
        self.hostname = "0.0.0.0"   # listen on all network interfaces
        self.port = port            # listen on the provided port
        self._socket = None         # initialize empty socket
        
        self.telemetry_queue = telemetry_queue
        

    def _setup_socket(self):
        """
        Create and bind the UDP socket.
        """
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.settimeout(1.0)  # add listening timeout in seconds
        self._socket.bind((self.hostname, self.port))
        logger.info(f"Telemetry listener bound on {self.hostname}:{self.port}")

        # Sets a timeout of 1.0 second on the socket. 
        # Any blocking socket operation (like recvfrom()) will wait for a maximum of 1.0 second for data to become available. 
        # If no data arrives within that 1.0 second, the recvfrom() call will not block indefinitely; 
        # instead, it will raise a socket.timeout exception


    def run(self):
        """
        Main process loop - binds to UDP socket and listens for telemetry messages.
        """
        try:
            self._setup_socket()
            self._listen_loop()
        except Exception as e:
            logger.error(f"Error in StreamTelemetryListener process: {e}")
        finally:
            # terminate process if failed to setup socket or listening is done
            self._terminate_process()
    

    def _listen_loop(self):
        """
        Main listening loop for receiving telemetry messages.
        """
        while not self._stop_event.is_set():
            
            try:
                # receive the data
                data, addr = self._socket.recvfrom(4096)
                # Decode and parse JSON
                telemetry = json.loads(data.decode('utf-8'))
                queue_object = TelemetryQueueObject(telemetry=telemetry, timestamp=time.time())
                self.telemetry_queue.put(queue_object, timeout=1.0)
                logger.debug(f"Captured telemetry from {addr}: {len(data)} bytes")
            
            except socket.timeout:
                # Timeout allows us to check stop_event periodically
                logger.error("socket timed-out, continuing to try to receive ...")
            except json.JSONDecodeError as e:
                logger.error(f"Received invalid JSON")
                time.sleep(0.1)
            except mp.queues.Full as e: # Catch Full specifically
                logger.error(f"Failed to put telemetry in queue: Queue is full. Consumer too slow or stopped?. Continuing to listen ...")
                time.sleep(0.1)
            except Exception as e:
                # if any other exception is raised, check wheter to stop or not
                if not self._stop_event.is_set():
                    logger.error(f"Error in readinf and processing telemetry: {e}")
                    time.sleep(0.1)

    def _terminate_process(self) -> None:
        """
        Send termination signal and clean up resources.
        """

        if self._socket:
            self._socket.close()
            self._socket = None
        logger.info("Telemetry listener cleaned up")

    def is_running(self) -> bool:
        """
        Check if the process is currently running.
        
        Returns:
            bool: True if process is alive and not stopped
        """
        return self.is_alive() and not self._stop_event.is_set()

    def stop(self) -> None:
        """
        Signal the process to stop gracefully.
        """
        logger.info("Stopping telemetry listener...")
        self._stop_event.set()

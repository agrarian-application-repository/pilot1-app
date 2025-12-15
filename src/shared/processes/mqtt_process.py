import asyncio
import logging
import ssl  # Needed for creating SSL context parameters
from aiomqtt.exceptions import MqttError
from aiomqtt import Client, TLSParameters
from time import time
from src.shared.processes.messages import TelemetryQueueObject
from multiprocessing import Queue as MPQueue
from multiprocessing import Process, Event

# ================================================================
logger = logging.getLogger("main.mqtt_telemetry_listener")

if not logger.handlers:  # Avoid duplicate handlers
    video_handler = logging.FileHandler('/app/logs/mqtt_telemetry_listener.log', mode='w')
    video_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(video_handler)
    logger.setLevel(logging.DEBUG)
# ================================================================

# --- Configuration for TLS ---
# NOTE: Replace with your actual DJI Cloud API details.
# The DJI broker will likely require authentication and specific CA certs.
BROKER_HOST = "test.mosquitto.org"  # Example public broker that supports TLS
BROKER_PORT = 8883  # Standard secure MQTTS port

# If the DJI broker requires a specific root certificate, download it and
# specify its path here. If using a public broker with a standard certificate,
# setting 'cert_reqs' to CERT_REQUIRED is often enough, but you may need 'ca_certs'.
CA_CERTS_FILE = None  # Replace with '/path/to/your/ca.crt' if needed
CERT_VALIDATION = ssl.CERT_REQUIRED  # Ensure the broker's certificate is valid

RECONNECT_DELAY = 5  # Seconds to wait before attempting reconnection

USERNAME = "user"
PASSWORD = "passw"

TOPICS_TO_SUBSCRIBE = [
    "telemetry/drone/latitude",
    "telemetry/drone/longitude",
    "telemetry/drone/rel_alt",
    "telemetry/drone/gb_yaw",
]
QOS_LEVEL = 1
# QoS 0 (At most once): no acknowledgment from the receiver
# QoS 1 (At least once):  ensures that messages are delivered at least once by requiring a PUBACK acknowledgment
# QoS 2 (Exactly once): guarantees that each message is delivered exactly once by using a four-step handshake (PUBLISH, PUBREC, PUBREL, PUBCOMP)

TEMPLATE_TELEMETRY = {
    "latitude": 44.414622942776454,
    "longitude": 8.880484631296774,
    "rel_alt": 40.0,
    "gb_yaw": 270.0,
}

TOPICS_TO_TELEMETRY_MAPPING = {
    "telemetry/drone/latitude": "latitude",
    "telemetry/drone/longitude": "longitude",
    "telemetry/drone/rel_alt": "rel_alt",
    "telemetry/drone/gb_yaw": "gb_yaw",
}


class MqttCollectorProcess(Process):
    """
    Dedicated process for collecting high-frequency MQTT telemetry data.
    """

    def __init__(self, telemetry_queue: MPQueue, stop_event: Event):
        # Initialize the base Process class
        super().__init__(name="MQTT_Collector_Process")

        # Shared output queue. Next process will read from this
        self.telemetry_queue = telemetry_queue

        # Shared stop event. Allows to stop all processes at the same time
        self.stop_event = stop_event

        # Configuration for the MQTT client (static)
        self.client_id = f"mqtts-subscriber-{self.pid or 'init'}"  # PID is available after start()
        self.tls_params = TLSParameters(ca_certs=CA_CERTS_FILE, cert_reqs=CERT_VALIDATION)
        self.telemetry_state = TEMPLATE_TELEMETRY.copy()

    def _create_mqtt_client(self):
        """Helper to create and configure the aiomqtt client."""
        return Client(
            hostname=BROKER_HOST,
            port=BROKER_PORT,
            identifier=self.client_id,
            tls_params=self.tls_params,
            username=USERNAME,
            password=PASSWORD,
            max_queued_incoming_messages=2_000,
        )

    async def _mqtt_subscriber_worker(self):
        """
        Asynchronous worker logic for connecting, subscribing, and message processing.
        """

        # Use a non-blocking check to see if we should stop.
        while not self.stop_event.is_set():

            # Create MQTT client.
            # The client is recreated as a clean object at every disconnection.
            # Safer: discard potentially dirty old object states from reusing the same old client
            client = self._create_mqtt_client()

            try:
                logger.info(f"Attempting secure connection (TLS) to {BROKER_HOST}:{BROKER_PORT}")

                # The 'async with' context manager handles connection/disconnection
                async with client:
                    logger.info("Secure connection successful. Subscribing to topics...")

                    # Subscribe to all topics
                    for topic_to_subscribe_to in TOPICS_TO_SUBSCRIBE:
                        await client.subscribe(topic=topic_to_subscribe_to, qos_level=QOS_LEVEL)
                        logger.info(f"Subscribed to topic '{topic_to_subscribe_to}' with QoS level '{QOS_LEVEL}'")

                    # Process messages using an async iterator
                    # We wrap the iterator with a timeout/task to check the stop_event

                    # The message loop itself is placed in a separate task
                    message_task = asyncio.create_task(self._process_messages(client))

                    # Wait for the message task (listener) to complete or be cancelled (via connection loss/stop signal)
                    await message_task

            except MqttError as e:
                # Handles MQTT-specific errors (network disconnect, broker kicks client)
                logger.error(f"MQTT Error caught: {e}. Reconnecting in {RECONNECT_DELAY}s...")
                await asyncio.sleep(RECONNECT_DELAY)

            except ConnectionRefusedError:
                # Handles initial connection failures (broker port closed/down)
                logger.error(f"Connection refused. Retrying in {RECONNECT_DELAY}s...")
                await asyncio.sleep(RECONNECT_DELAY)

            except ssl.SSLError as e:
                # Handles errors during the TLS handshake (e.g., invalid certificate)
                logger.error(
                    f"TLS/SSL Error: {e}. Check certificate paths and broker setup. Retrying in {RECONNECT_DELAY}s...")
                await asyncio.sleep(RECONNECT_DELAY)

            except Exception as e:
                # Catch all other unexpected errors
                logger.error(f"An unexpected error occurred: {e}. Retrying in {RECONNECT_DELAY}s...")
                await asyncio.sleep(RECONNECT_DELAY)

    async def _process_messages(self, client):
        """
        Inner loop for processing messages with concurrent stop check.
        """
        # this for exists with a MQTTException when the connection is broken
        async for message in client.messages:

            # Check for the stop signal quickly after receiving a message
            if self.stop_event.is_set():
                break

            topic = message.topic.value
            try:
                # 1. Decode and Update State
                payload = message.payload.decode()
                telemetry_key = TOPICS_TO_TELEMETRY_MAPPING[topic]
                self.telemetry_state[telemetry_key] = payload

                # 2. Put Data on Queue
                msg = TelemetryQueueObject(
                    telemetry=self.telemetry_state.copy(),
                    timestamp=time()
                )
                # Use put() without timeout/nowait. The multiprocessing queue is robust.
                self.telemetry_queue.put(msg)
            except Exception as e:
                logger.error(f"Error processing message from topic {topic}: {e}")
                # Continue to next message instead of crashing the process when the error
                # only has to do with the message content, but the connection is still up and
                # the stop signal has not been set

    def run(self):
        """
        The method executed when the Process is started.
        It must contain the setup for the asyncio event loop.
        """
        logger.info(f"Collector Process started. PID: {self.pid}")
        try:
            # Start the asyncio event loop and run the main worker coroutine
            asyncio.run(self._mqtt_subscriber_worker())
        except Exception as e:
            # If the process crashes outside of the worker loop
            logger.critical(f"Collector Process crashed fatally: {e}")
        finally:
            logger.info(f"Collector Process {self.pid} shutting down cleanly.")


# --- Main Application Example ---

if __name__ == "__main__":

    # 1. Setup Logging for Multiprocessing
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - (%(processName)s) - %(message)s')

    # 2. Initialize Shared Objects
    telemetry_queue = MPQueue()
    stop_signal = Event()

    # 3. Instantiate the Process
    collector = MqttCollectorProcess(telemetry_queue, stop_signal)

    # 4. Start the Process
    collector.start()
    logger.info(f"Collector Process initiated (PID: {collector.pid}).")

    # 5. Example of Main Loop (e.g., retrieving data)
    try:
        while True:
            # Main thread does other work or pulls data from the queue
            if not telemetry_queue.empty():
                latest_data = telemetry_queue.get()
                logger.info(
                    f"Main App received update at {latest_data.timestamp}. Lat: {latest_data.telemetry['latitude']}")

            # Sleep briefly to avoid 100% CPU usage
            asyncio.sleep(0.1)

    except KeyboardInterrupt:
        logger.warning("Keyboard Interrupt received. Initiating graceful shutdown...")
    finally:
        # 6. Graceful Shutdown
        stop_signal.set()  # Set the event to signal the child process to stop its while True loop
        collector.join(timeout=10)  # Wait up to 10 seconds for the process to finish

        if collector.is_alive():
            logger.error("Process did not shut down cleanly. Forcing termination.")
            collector.terminate()

        logger.info("Application successfully terminated.")
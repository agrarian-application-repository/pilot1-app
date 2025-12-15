import asyncio
import logging
import ssl  # Needed for creating SSL context parameters
from aiomqtt.exceptions import MqttError
from aiomqtt import Client, TLSParameters
from time import time
from src.shared.processes.messages import TelemetryQueueObject
from multiprocessing import Queue as MPQueue

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


# --- Main Asynchronous Client Logic ---
async def mqtt_subscriber_worker(telemetry_queue: MPQueue):
    """
    Asynchronous worker using TLS for secure connection.
    Supports auto-reconnection.
    """
    client_id = f"mqtts-subscriber-{hash(asyncio.current_task())}"

    # Define TLS Parameters
    tls_params = TLSParameters(
        ca_certs=CA_CERTS_FILE,
        cert_reqs=CERT_VALIDATION,
    )

    # Create async MQTT client
    client = Client(
        hostname=BROKER_HOST,
        port=BROKER_PORT,
        identifier=client_id,
        tls_params=tls_params,
        username=USERNAME,
        password=PASSWORD,
        max_queued_incoming_messages=2_000,
    )

    telemetry_state = TEMPLATE_TELEMETRY.copy()

    while True:

        try:
            logger.info(f"Attempting secure connection (TLS) to {BROKER_HOST}:{BROKER_PORT}")

            async with client:
                logger.info("Secure connection successful. Subscribing to topics...")
                # Subscribe to all topics
                for topic_to_subscribe_to in TOPICS_TO_SUBSCRIBE:
                    await client.subscribe(topic=topic_to_subscribe_to, qos_level=QOS_LEVEL)
                    logger.info(f"Subscribed to topic '{topic_to_subscribe_to}' with QoS level '{QOS_LEVEL}'")

                # Process messages using an async iterator
                async for message in client.messages:
                    # 1. parse topic
                    topic = message.topic.value
                    payload = message.payload.decode()
                    logger.debug(f"[{topic}]: {payload}...")
                    # 2. update telemetry dict
                    telemetry_key = TOPICS_TO_TELEMETRY_MAPPING[topic]
                    telemetry_state[telemetry_key] = payload
                    # 3. save on multiprocessing queue for next process to retrieve data
                    msg = TelemetryQueueObject(telemetry=telemetry_state.copy(), timestamp=time())
                    telemetry_queue.put(msg)    # TODO: fix for multiprocessing queue (nowait?, wait?)

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


# --- Graceful Termination Setup ---

async def main():
    """
    The main asyncio entry point. Sets up the subscriber worker and handles termination signals.
    """
    logger.info("Starting aiomqtt TLS subscriber...")
    worker_task = asyncio.create_task(mqtt_subscriber_worker())

    # cancel task:
    # worker_task.cancel()

    # Handling Termination (Ctrl+C)
    try:
        await worker_task
    except asyncio.CancelledError:
        logger.info("Main worker cancelled. Shutting down gracefully...")

    await asyncio.sleep(0.5)
    logger.info("Application terminated.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Keyboard Interrupt received. Stopping application.")
    except RuntimeError as e:
        # Handling for environments where loop is already running
        if "cannot run" in str(e):
            asyncio.create_task(main())
        else:
            raise

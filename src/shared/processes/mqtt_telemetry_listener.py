import asyncio
import logging
import ssl  # Needed for creating SSL context parameters
from queue import Full as QueueFullException
from aiomqtt.exceptions import MqttError
from aiomqtt import Client, TLSParameters
from time import sleep, time
from src.shared.processes.messages import TelemetryQueueObject
import multiprocessing as mp
from src.shared.processes.constants import *
from src.shared.processes.consumer import Consumer
from typing import Optional

# ================================================================
logger = logging.getLogger("main.mqtt_telemetry_listener")

if not logger.handlers:  # Avoid duplicate handlers
    video_handler = logging.FileHandler('./mqtt_telemetry_listener.log', mode='w')
    video_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(video_handler)
    logger.setLevel(logging.DEBUG)
# ================================================================


class MqttCollectorProcess(mp.Process):
    """
    Dedicated process for collecting high-frequency MQTT telemetry data.
    """

    def __init__(
            self, 
            telemetry_queue: mp.Queue, 
            stop_event: mp.Event,                       # stop event used to stop gracefully
            error_event: mp.Event,                      # error event used to stop gracefully on processing error
            protocol: str = MQTT,
            broker_host: str = TELEMETRY_LISTENER_HOST,    # Example public broker that supports TLS
            broker_port: int = TELEMETRY_LISTENER_PORT,              # Standard secure MQTTS port
            username: Optional[str] = None,
            password: Optional[str] = None,
            qos_level: int = TELEMETRY_LISTENER_QOS_LEVEL,
            max_msg_wait: float = TELEMETRY_LISTENER_MSG_WAIT_TIMEOUT,
            reconnection_delay: float = TELEMETRY_LISTENER_RECONNECT_DELAY,
            ca_certs_file_path: str = "certificates/mqtt",
            cert_validation: int = TELEMETRY_LISTENER_CERT_VALIDATION,
            max_incoming_messages: int = TELEMETRY_LISTENER_MAX_INCOMING_MESSAGES,
    ):

        # Initialize the base Process class
        super().__init__(name="MQTT_Collector_Process")

        # Shared output queue. Next process will read from this
        self.telemetry_queue = telemetry_queue

        # Shared stop event.
        # Allows to stop all processes at the same time
        # (sent by video reading process on correct termination)
        self.stop_event = stop_event

        # Shared error event.
        # Allows to stop all processes at the same time
        # (sent by any process terminating unexpectedly due to error, all other processes should stop)
        self.error_event = error_event
        
        self.broker_host = broker_host
        self.broker_port = broker_port
        
        self.username = username
        self.password = password

        self.qos_level = qos_level

        self.reconnection_delay = reconnection_delay

        self.max_msg_wait = max_msg_wait
        self.max_incoming_messages = max_incoming_messages

        self.work_finished = mp.Event()

        # Configuration for the MQTT client (static)
        self.client_id = None  # placeholder
        self.tls_params = None
        
        if protocol == MQTTS:
            self.tls_params = TLSParameters(
                ca_certs=ca_certs_file_path,
                cert_reqs=cert_validation,
            )

        self.telemetry_state = TELEMETRY_LISTENER_TEMPLATE_TELEMETRY.copy()

    def _create_mqtt_client(self):
        """Helper to create and configure the aiomqtt client."""
        return Client(
            hostname=self.broker_host,
            port=self.broker_port,
            identifier=self.client_id,
            tls_params=self.tls_params,
            username=self.username,
            password=self.password,
            max_queued_incoming_messages=self.max_incoming_messages,
        )

    async def _mqtt_subscriber_worker(self):
        """
        Asynchronous worker logic for connecting, subscribing, and message processing.
        """

        reconnect_str = f"Retrying to connect in {self.reconnection_delay} seconds..."

        # Use a non-blocking check to see if we should stop.
        while not (self.stop_event.is_set() or self.error_event.is_set()):

            # Create MQTT client.
            # The client is recreated as a clean object at every disconnection.
            # Safer: discard potentially dirty old object states from reusing the same old client
            client = self._create_mqtt_client()

            try:
                logger.info(f"Attempting secure connection (TLS) to {self.broker_host}:{self.broker_port}")

                # The 'async with' context manager handles connection/disconnection
                async with client:
                    logger.info("Secure connection successful. Subscribing to topics...")

                    # Subscribe to all topics
                    for topic_to_subscribe_to in TELEMETRY_LISTENER_TOPICS_TO_SUBSCRIBE:
                        await client.subscribe(topic=topic_to_subscribe_to, qos=self.qos_level)
                        logger.info(f"Subscribed to topic '{topic_to_subscribe_to}' with QoS level '{self.qos_level}'")

                    # Process messages using an async iterator
                    # We wrap the iterator with a timeout/task to check the stop_event and error_event
                    # The message loop itself is placed in a separate task
                    message_task = asyncio.create_task(self._process_messages(client))

                    # Wait for the message task (listener) to stop, either via:
                    # - client cleanly disconnected, continue this loop for reconnection
                    # - exception: connection fails
                    # - stop_event: stop receiving because the application is shutting down at job done
                    # - error_event: stop receiving because one of the downstream processing process has failed
                    # while none of these happens, the precess_messages task continues collecting data, and we do not
                    # reach this point (task is blocking)
                    await message_task

            except MqttError as e:
                # Handles MQTT-specific errors (network disconnect, broker kicks client)
                logger.error(f"MQTT Error caught: {e}. {reconnect_str}")
                await asyncio.sleep(self.reconnection_delay)

            except ConnectionRefusedError:
                # Handles initial connection failures (broker port closed/down)
                logger.error(f"Connection refused. {reconnect_str}")
                await asyncio.sleep(self.reconnection_delay)

            except ssl.SSLError as e:
                # Handles errors during the TLS handshake (e.g., invalid certificate)
                logger.error(
                    f"TLS/SSL Error: {e}. Check certificate paths and broker setup. {reconnect_str}")
                await asyncio.sleep(self.reconnection_delay)

            except Exception as e:
                # Catch all other unexpected errors
                logger.error(f"An unexpected error occurred: {e}. {reconnect_str}")
                await asyncio.sleep(self.reconnection_delay)

    async def _process_messages(self, client):
        """
        Inner loop for processing messages with concurrent stop check.
        """
        while not (self.stop_event.is_set() or self.error_event.is_set()):

            # wait for message for of timeout second
            # This doesn't catch MqttError
            # It simply allows to periodically check the stopping conditions
            try:
                async with asyncio.timeout(self.max_msg_wait):
                    # Get the next message from the iterator manually
                    message = await anext(client.messages)
                    logger.debug(f"Message received on topic {message.topic.value}")
            except asyncio.TimeoutError:
                # No message arrived within timeout, loop back to check halting events
                logger.warning(f"No messages received for {self.max_msg_wait} seconds. Continuing to listen ...")
                continue
            except StopAsyncIteration:
                # The iterator closed (Connection closed by broker)
                # We let this break the loop so the worker handles reconnection
                logger.warning(f"Client disconnected, attempting to reconnect ...")
                break

            # ---------- message received -------------

            topic = message.topic.value
            try:
                # 1. Decode and Update State
                payload = message.payload.decode()
                telemetry_key = TELEMETRY_LISTENER_TOPICS_TO_TELEMETRY_MAPPING.get(topic)
                if not telemetry_key:
                    logger.warning(f"Received a message from an unexpected topic {topic}. Skipped.")
                    continue  # Ignore topics we don't map
                self.telemetry_state[telemetry_key] = payload
            except Exception as e:
                logger.error(f"Error processing message from topic {topic}: {e}. Continuing to listen.")
                continue
                # Continue to next message instead of crashing the process when the error
                # only has to do with the message content, but the connection is still up and
                # the stop signal has not been set

            # 2. Create message
            msg = TelemetryQueueObject(
                telemetry=self.telemetry_state.copy(),
                timestamp=time()
            )

            # 3. Put message on queue
            try:
                # nowait since many updates per second are expected, don't want to stay blocked
                self.telemetry_queue.put_nowait(msg)

            # if the message is lost, so be it.
            # Just log the warning, then continue on with the listening loop
            except QueueFullException:
                logger.warning(
                    f"Output queue is full. Consumer too slow? "
                    f"Telemetry dropped. Continuing ..."
                )
            except Exception as e:
                logger.error(
                    f"Failed to put telemetry value on queue: {e}. "
                    f"Telemetry dropped. Continuing ..."
                )

    def run(self):
        """
        The method executed when the Process is started.
        It must contain the setup for the asyncio event loop.
        """
        self.client_id = f"mqtts-subscriber-{self.pid or 'init'}"  # PID is available after start()
        logger.info(f"MQTT Telemetry Listener Process started. PID: {self.pid}")

        try:
            # Start the asyncio event loop and run the main worker coroutine
            asyncio.run(self._mqtt_subscriber_worker())
        except Exception as e:
            # If the process crashes outside of the worker loop
            logger.critical(
                f"MQTT Telemetry listener Process crashed fatally: {e}."
                f"The application will continue to run without telemetry. "
            )
            # Processing can continue without telemetry
            # ==> DO NOT set the error event which would halt all other processes in their track
        finally:
            logger.info(
                f"Collector Process {self.pid} shutting down cleanly after halting event has been received. "
                f"Stop event: {self.stop_event.is_set()}. "
                f"Error event: {self.error_event.is_set()}."
            )
            self.work_finished.set()


# --- Main Application Example ---

if __name__ == "__main__":

    broker_host = "0.0.0.0"
    broker_port = MQTT_PORT

    VSLOW = 1
    SLOW = 10
    FAST = 50
    REAL = 30
    FREAL = 40

    QUEUE_SIZE = 20

    telemetry_queue = mp.Queue(maxsize=QUEUE_SIZE)
    stop_event = mp.Event()
    error_event = mp.Event()

    collector = MqttCollectorProcess(telemetry_queue, stop_event, error_event, broker_host=broker_host, broker_port=broker_port, cert_validation=None)
    
    consumer = Consumer(telemetry_queue, error_event, frequency_hz=FAST)

    print("CONSUMERS STARTED")
    consumer.start()

    sleep(3)

    print("COLLECTOR STARTED")
    collector.start()

    sleep(5)

    # option 1
    print("STOP EVENT")
    stop_event.set()
    telemetry_queue.put(POISON_PILL)  # ensure consumer knows it must stop (collector does nnot provide POISON PILL)         
    
    # option2
    #print("ERROR EVENT")
    #error_event.set()              

    processes = [collector, consumer]

    while True:

        # Check if everyone has finished their logic
        all_finished = all(p.work_finished.is_set() for p in processes)

        # Check if an error occurred anywhere
        error_occurred = error_event.is_set()

        if all_finished or error_occurred:
            if error_occurred:
                print("[Main] Error detected. Terminating chain.")
            else:
                print("[Main] All processes finished logic. Cleaning up.")
            break

        sleep(0.5)

    print(f"[Main] Granting 5s for all processed to cleanly conclude their processing.")
    sleep(5.0)
    # The Sweep: Force everyone to join or die
    for p in processes:
        # If the logic is finished but the process is still 'alive',
        # it is 100% stuck in the queue feeder thread.
        if p.is_alive():
            print(f"[Main] {p.name} is hanging in cleanup. Work Completed: {p.work_finished.is_set()}. Terminating.")
            p.terminate()

        p.join()
        print(f"[Main] {p.name} joined.")

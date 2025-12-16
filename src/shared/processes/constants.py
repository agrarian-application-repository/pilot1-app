import ssl  # Needed for creating SSL context parameters
import cv2
# -------------------------- GENERAL READER --------------------------

# value of the poison pill to stop following processes
POISON_PILL = "HALT"
POISON_PILL_TIMEOUT = 5.0

# how long to wait to get message from input queue
QUEUE_GET_TIMEOUT = 0.01

# image downsampling interpolation
DOWNSAMPLING_MODE = cv2.INTER_LINEAR

# image upsampling interpolation
UPSAMPLING_MODE = cv2.INTER_LINEAR

# ------------------------ VIDEO READING ----------


# VIDEO_STREAM_URL = "rtmp://<server>[:port]/<app>/<stream_key>"
# VIDEO_STREAM_URL = "rtmps://<server>[:port]/<app>/<stream_key>"
# VIDEO_STREAM_URL = "rtsp://[user[:password]@]host[:port]/path"
# VIDEO_STREAM_URL = "rtsps://[user[:password]@]host[:port]/path"

VIDEO_STREAM_READER_CONNECTION_OPEN_TIMEOUT_S = 5.0
VIDEO_STREAM_READER_RECONNECT_DELAY = 5.0
VIDEO_STREAM_READER_MAX_CONSECUTIVE_CONNECTION_FAILURES = 5

VIDEO_STREAM_READER_FRAME_READ_TIMEOUT_S = 0.1
VIDEO_STREAM_READER_FRAME_RETRY_DELAY = 0.1
VIDEO_STREAM_READER_FRAME_MAX_CONSECUTIVE_FAILURES = 50

VIDEO_STREAM_READER_EXPECTED_ASPECT_RATIO = 16.0/9.0
VIDEO_STREAM_READER_PROCESSING_SHAPE = (1280, 720)  # (W,H)

VIDEO_STREAM_READER_BUFFER_SIZE = 1

VIDEO_READING_OUT_QUEUE_PUT_TIMEOUT = 0.02     # block for up to 20 ms to put data in output queue


# -------------------------- MQTT READER --------------------------

# Standard MQTTS port
MQTTS_PORT = 8883

# If the DJI broker requires a specific root certificate, download it and
# specify its path here. If using a public broker with a standard certificate,
# setting 'cert_reqs' to CERT_REQUIRED is often enough, but you may need 'ca_certs'.
MQTT_CERT_VALIDATION = ssl.CERT_REQUIRED  # Ensure the broker's certificate is valid

# Seconds to wait before attempting reconnection
MQTT_RECONNECT_DELAY = 5.0

MQTT_TOPICS_TO_SUBSCRIBE = [
    "telemetry/drone/latitude",
    "telemetry/drone/longitude",
    "telemetry/drone/rel_alt",
    "telemetry/drone/gb_yaw",
]
MQTT_QOS_LEVEL = 1
# QoS 0 (At most once): no acknowledgment from the receiver
# QoS 1 (At least once):  ensures that messages are delivered at least once by requiring a PUBACK acknowledgment
# QoS 2 (Exactly once): guarantees that each message is delivered exactly once by using a four-step handshake
# (PUBLISH, PUBREC, PUBREL, PUBCOMP)


MQTT_TOPICS_TO_TELEMETRY_MAPPING = {
    "telemetry/drone/latitude": "latitude",
    "telemetry/drone/longitude": "longitude",
    "telemetry/drone/rel_alt": "rel_alt",
    "telemetry/drone/gb_yaw": "gb_yaw",
}

TEMPLATE_TELEMETRY = {
    "latitude": 44.414622942776454,
    "longitude": 8.880484631296774,
    "rel_alt": 40.0,
    "gb_yaw": 270.0,
}

MQTT_MSG_WAIT_TIMEOUT = 1.0

# size of the input messages queue
MQTT_MAX_INCOMING_MESSAGES = 2_000


# -------------------------- ALERTS WRITER --------------------------

MAX_ALERTS_STORED = 20
WS_COMMON_PORT = 8765
WS_PORT = 80
WSS_PORT = 443
JPEG_COMPRESSION_QUALITY = 85
ALERTS_GET_TIMEOUT = 0.5

WS_MANAGER_QUEUE_WAIT_TIMEOUT = 0.02
WS_MANAGER_STOP_EVENT_CHECK_PERIOD = 0.1
WS_MANAGER_THREAD_CLOSE_TIMEOUT = 5.0

#DB_USER = os.getenv("DB_USER", "default_user")
#DB_PASS = os.getenv("DB_PASS", "")     # TODO handle special characters
#DB_HOST = os.getenv("DB_HOST", "localhost")
#DB_NAME = os.getenv("DB_NAME", "alert_system")
#db_url = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"

DB_POOL_SIZE = 5
DB_MAX_OVERFLOW = 10

DB_MANAGER_QUEUE_WAIT_TIMEOUT = 0.1
DB_MANAGER_THREAD_CLOSE_TIMEOUT = 5.0

DB_MANAGER_QUEUE_SIZE = 100




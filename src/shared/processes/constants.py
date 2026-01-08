import ssl  # Needed for creating SSL context parameters
import cv2


# REQUEST TO USER (CREDENTIALS):
# - AGRARIAN_USERNAME
# - AGRARIAN_PASSWORD
# - DB_USERNAME
# - DB_PASSWORD
# - MQTT_USERNAME
# - MQTT_PASSWORD
# - MQTT_HOST
# - RTMP_STREAM_KEY
# - RTMP_HOST

# REQUEST TO USERS (FILES, mount)
# - MQTT_CERT
# - DEM
# - DEM_MASK
# - DRONE_CONFIG_FILE
# - INPUT_CONFIG_FILE

# WHAT THE USER WILL GET TO KNOW:
# IP/PORT OF WEBCOKET SERVER
# IP/PORT TO RETRIEVE THE STREAM

# -------------------------- GENERAL --------------------------

FPS = 30
TPS = 50

# value of the poison pill to stop following processes
POISON_PILL = "HALT"
POISON_PILL_TIMEOUT = 5.0                                               # 5.0 s

# how long to wait to get message from input queue
QUEUE_GET_TIMEOUT = 0.01                                                # 10 ms

# image downsampling interpolation
DOWNSAMPLING_MODE = cv2.INTER_LINEAR

# image upsampling interpolation
UPSAMPLING_MODE = cv2.INTER_LINEAR

# -------------------------- QUEUES SIZES --------------------------

SECONDS_OF_VIDEO_FRAMES = FPS * 3
SECONDS_OF_TELEMETRY = TPS * 3

FRAME_2_FRAMETELEM_COMBINER = SECONDS_OF_VIDEO_FRAMES               # 3 seconds of video
TELEMETRY_2_FRAMETELEM_COMBINER = SECONDS_OF_TELEMETRY * 4          # 4 topics for 3 seconds of telemetry
FRAMETELEM_COMBINER_2_MODEL = SECONDS_OF_VIDEO_FRAMES
MODEL_2_RESULT = SECONDS_OF_VIDEO_FRAMES
RESULT_2_ANNOTATION = SECONDS_OF_VIDEO_FRAMES
ANNOTATION_2_STREAM = SECONDS_OF_VIDEO_FRAMES                       # 3 seconds of video
ANNOTATION_2_ALERT = 15                                             # 15 alerts
STREAM_2_UPLOAD = 3                                                 # url + stop + 1 spare

# ------------------------ VIDEO READING ----------

# VIDEO_STREAM_URL = "rtmp://<server>[:port]/<app>/<stream_key>"
# VIDEO_STREAM_URL = "rtmps://<server>[:port]/<app>/<stream_key>"
# VIDEO_STREAM_URL = "rtsp://[user[:password]@]host[:port]/path"
# VIDEO_STREAM_URL = "rtsps://[user[:password]@]host[:port]/path"

VIDEO_STREAM_READER_CONNECTION_OPEN_TIMEOUT_S = 5.0
VIDEO_STREAM_READER_RECONNECT_DELAY = 5.0
VIDEO_STREAM_READER_MAX_CONSECUTIVE_CONNECTION_FAILURES = 5

VIDEO_STREAM_READER_FRAME_READ_TIMEOUT_S = 0.05                         # 50 ms
VIDEO_STREAM_READER_FRAME_RETRY_DELAY = 0.05                            # 50 ms
VIDEO_STREAM_READER_FRAME_MAX_CONSECUTIVE_FAILURES = 30

VIDEO_STREAM_READER_EXPECTED_ASPECT_RATIO = 16.0/9.0
VIDEO_STREAM_READER_PROCESSING_SHAPE = (1280, 720)  # (W,H)

VIDEO_STREAM_READER_BUFFER_SIZE = 1

VIDEO_STREAM_READER_QUEUE_PUT_TIMEOUT = 0.02                              # 20 ms


# -------------------------- MQTT READER --------------------------

# testing host
MQTT_HOST = "test.mosquitto.org"

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

# max thread blocking message wait, after this, check again wheter a stop signal has been received
MQTT_MSG_WAIT_TIMEOUT = 1.0

# size of the input messages queue
MQTT_MAX_INCOMING_MESSAGES = 2_000


# -------------------------------------------------------------------
# -------------------------- FRAME + TELEMETRY COMBINING ------------
# -------------------------------------------------------------------
FRAMETELCOMB_MAX_TELEM_BUFFER_SIZE = TELEMETRY_2_FRAMETELEM_COMBINER * 2    # double process input queue
FRAMETELCOMB_MAX_TIME_DIFF = 0.15                   # 150 ms
FRAMETELCOMB_QUEUE_GET_TIMEOUT = 0.01               # 10 ms
FRAMETELCOMB_QUEUE_PUT_MAX_RETRIES = 3              # 3
FRAMETELCOMB_QUEUE_PUT_BACKOFF = 0.005              # 5 ms  (15 ms over 3 retries)

# -------------------------------------------------------------------
# -------------------------- MODELS & ANNOTATIONS -------------------
# -------------------------------------------------------------------
MODELS_QUEUE_GET_TIMEOUT = 0.02                     # 20 ms
MODELS_QUEUE_PUT_TIMEOUT = 0.02                     # 20 ms

DANGER_MAX_CONSECUTIVE_FAILURES = 5

ANNOTATION_QUEUE_GET_TIMEOUT = 0.02
ANNOTATION_QUEUE_PUT_TIMEOUT = 0.02
ANNOTATION_MAX_CONSECUTIVE_FAILURES = 5
ANNOTATION_MAX_PUT_ALERT_CONSECUTIVE_FAILURES = 3
ANNOTATION_MAX_PUT_VIDEO_CONSECUTIVE_FAILURES = FPS

# -------------------------------------------------------------------
# -------------------------- ALERTS WRITER --------------------------
# -------------------------------------------------------------------

ALERTS_GET_TIMEOUT = 0.1                                # 100 ms

ALERTS_MAX_CONSECUTIVE_FAILURES = 5
ALERTS_JPEG_COMPRESSION_QUALITY = 85

# -------------------------- ALERTS WS --------------------------

WS_COMMON_PORT = 8765
WS_PORT = 80
WSS_PORT = 443

WS_MANAGER_BROADCAST_TIMEOUT = 2.0
WS_MANAGER_PING_INTERVAL = 5.0                          # 5.0 s
WS_MANAGER_PING_TIMEOUT = 20.0                          # 20.0 s
WS_MANAGER_THREAD_CLOSE_TIMEOUT = 10.0                  # 10.0 s

# -------------------------- ALERTS DB --------------------------

#DB_USER = os.getenv("DB_USER", "default_user")
#DB_PASS = os.getenv("DB_PASS", "")     # --> TODO: handle special characters
#DB_HOST = os.getenv("DB_HOST", "localhost")
#DB_NAME = os.getenv("DB_NAME", "alert_system")
#db_url = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"

DB_MANAGER_QUEUE_SIZE = 20

DB_MANAGER_POOL_SIZE = 5
DB_MANAGER_MAX_OVERFLOW = 10

DB_MANAGER_QUEUE_WAIT_TIMEOUT = 0.1                     # 100 ms
DB_MANAGER_THREAD_CLOSE_TIMEOUT = 10.0                  # 10.0 s


# -------------------------------------------------------------------
# -------------------------- OUT VIDEO WRITER --------------------------
# -------------------------------------------------------------------

VIDEO_WRITER_FPS = FPS
VIDEO_GET_FRAME_TIMEOUT = 0.01                              # 10 ms

# ------------------------- OUT VIDEO STREAM  --------------------------

VIDEO_OUT_STREAM_QUEUE_MAX_SIZE = VIDEO_WRITER_FPS * 3      # 3 minutes
VIDEO_OUT_STREAM_QUEUE_GET_TIMEOUT = 0.01                   # 10 ms
VIDEO_OUT_STREAM_FFMPEG_STARTUP_TIMEOUT = 0.5               # 0.5 s
VIDEO_OUT_STREAM_FFMPEG_SHUTDOWN_TIMEOUT = 8.0              # 8.0 s
VIDEO_OUT_STREAM_STARTUP_TIMEOUT = 2.0                      # 2.0 s
VIDEO_OUT_STREAM_SHUTDOWN_TIMEOUT = 10.0                    # 2.0 s

# ------------------------- OUT VIDEO STORE  --------------------------

VIDEO_OUT_STORE_MAX_UPLOAD_RETRIES = 5
VIDEO_OUT_STORE_RETRY_BACKOFF_TIME = 10.0                   # 10 s

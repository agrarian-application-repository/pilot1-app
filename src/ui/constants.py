import os

# Application constants
HOST = "0.0.0.0"
MAX_ALERT_QUEUE_SIZE = 50
MAX_ALERTS_DISPAYED = 5
ALERT_BOX_TIMEDIFF = 5.0
RECONNECT_DELAY = 5.0
REFRESH_STATS = 1.0
REFRESH_ALERTS = 1.0

# Configurable parameters with environment variable fallbacks
TCP_PORT = int(os.getenv("TCP_PORT", "54321"))
MEDIAMTX_WEBRTC_URL = os.getenv("MEDIAMTX_WEBRTC_URL", "http://mediamtx:8889")
STREAM_NAME = os.getenv("STREAM_NAME", "annot")
STUN_SERVER = os.getenv("STUN_SERVER", "stun:stun.l.google.com:19302")
CONTAINER = bool(os.getenv("CONTAINER", ""))
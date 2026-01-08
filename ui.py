import streamlit as st
from collections import deque
import queue
import logging
import streamlit.components.v1 as components
import datetime
import os
from time import time
from src.ui.alert_receiver import AlertReceiver
from src.ui.video_player import get_video_player

# ================================================================
# Logging Configuration
# ================================================================
log_path = "./logs/ui.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_path, mode='w')]
)

logger = logging.getLogger("ui")


# ================================================================
# Application constants
# ================================================================

WEBRTC_HOST = os.getenv("WEBRTC_HOST", "0.0.0.0")
WEBRTC_PORT = int(os.getenv("WEBRTC_PORT", 8889))
WEBRTC_STREAM_NAME = os.getenv("WEBRTC_STREAM_NAME", "annot")
WEBRTC_STUN_SERVER = os.getenv("WEBRTC_STUN_SERVER", "stun:stun.l.google.com:19302")

WEBSOCKET_HOST = os.getenv("WEBSOCKET_HOST", "0.0.0.0")
WEBSOCKET_PORT = int(os.getenv("WEBSOCKET_PORT", 443))
WEBSOCKET_RECONNECTION_DELAY = int(os.getenv("WEBSOCKET_RECONNECTION_DELAY", 5))
WEBSOCKET_PING_INTERVAL = int(os.getenv("WEBSOCKET_PING_INTERVAL", 30))
WEBSOCKET_PING_TIMEOUT = int(os.getenv("WEBSOCKET_PING_TIMEOUT", 10))

ALERTS_REFRESH = float(os.getenv("ALERTS_REFRESH", 1.0))
ALERTS_BOX_COLOR_TIMEDIFF = float(os.getenv("ALERTS_BOX_COLOR_TIMEDIFF", 5.0))
ALERTS_MAX_QUEUE_SIZE = int(os.getenv("ALERTS_MAX_QUEUE_SIZE", 20))
ALERTS_MAX_DISPLAYED = int(os.getenv("ALERTS_MAX_DISPLAYED", 5))

LOGO = "assets/leonardo.png"
LOGO_WIDTH = int(os.getenv("LOGO_WIDTH", 200))
HTML_HEIGHT = int(os.getenv("HTML_HEIGHT", 600))
ALERT_HEIGHT = int(os.getenv("ALERT_HEIGHT", 600))

WEBRTC_URL = f"http://{WEBRTC_HOST}:{WEBRTC_PORT}"
WEBSOCKET_URL = f"ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}"


# ================================================================
# initialization
# ================================================================

def initialize_services(
        ws_host: str,
        ws_port: int,
        webrtc_host: str, 
        webrtc_port: int,
        webrtc_stream: str, 
        webrtc_stun: str,
) -> bool:
    """Initialize all services automatically"""
    
    initialized = False

    if 'services_initialized' not in st.session_state:
        
        st.session_state.services_initialized = True
        
        # Initialize session state
        st.session_state.alerts_display_dequeue = deque(maxlen=ALERTS_MAX_DISPLAYED)
        st.session_state.alerts_reception_queue = queue.Queue(maxsize=ALERTS_MAX_QUEUE_SIZE)
        st.session_state.total_alerts = 0
        st.session_state.last_alert_timestamp = None

        # Initialize WebRTC config
        webrtc_config = {
            'host': webrtc_host,
            'port': webrtc_port,
            'stream': webrtc_stream,
            'stun': webrtc_stun,
        }
        logger.info(f"WebRTC configured as follows: {webrtc_config}")
        
        # Start WEBSOCKET receiver
        st.session_state.websocket_receiver = AlertReceiver(
            host=ws_host,
            port=ws_port,
            shared_queue=st.session_state.alerts_reception_queue,
            reconnection_delay=WEBSOCKET_RECONNECTION_DELAY,
            ping_interval=WEBSOCKET_PING_INTERVAL,
            ping_timeout=WEBSOCKET_PING_TIMEOUT,
        )
        st.session_state.ws_thread = st.session_state.websocket_receiver.start()
        logger.info(
            f"Websocket client initialized and started. "
            f"Waiting for alerts from {WEBSOCKET_URL}"
        )

        initialized = True

    return initialized


# ================================================================
# Alert Processing Fragments
# ================================================================

@st.fragment(run_every=ALERTS_REFRESH)
def update_metrics():

    if st.session_state.last_alert_timestamp is None:
        st.success("No alerts received yet")

    else:
        current_time = time()
        logger.debug(f"current time: {current_time}")
        logger.debug(f"last alert time: {st.session_state.last_alert_timestamp}")

        # compute time difference in UTC time
        seconds_passed = int(current_time - st.session_state.last_alert_timestamp)
        logger.info(f"seconds passed: {seconds_passed}")
        minutes_passed = seconds_passed // 60
        if minutes_passed > 0:
            seconds_passed = seconds_passed % 60

        # Convert time to local timestamp for displaying
        last_alert_local = (
            datetime.datetime
                .fromtimestamp(st.session_state.last_alert_timestamp, tz=datetime.timezone.utc)
                .astimezone()
                .strftime('%H:%M:%S')
        )

        logger.info(last_alert_local)

        if seconds_passed < 60:
            text = f"Alert received {seconds_passed} seconds ago ({last_alert_local})"
        else:
            text = f"Alert received {minutes_passed}:{seconds_passed} minutes ago ({last_alert_local})"
        
        if seconds_passed > ALERTS_BOX_COLOR_TIMEDIFF:
            st.warning(text)    # yellow if older
        else:
            st.error(text)      # red if very recent
    
    st.metric("Total Alerts", st.session_state.total_alerts)
    st.metric("Displayed Alerts", len(st.session_state.alerts_display_dequeue))
        

@st.fragment(run_every=ALERTS_REFRESH)
def process_alerts():
    """Process new alerts in a fragment that runs independently"""
    # Process new alerts
    try:
        while not st.session_state.alerts_reception_queue.empty():
            alert = st.session_state.alerts_reception_queue.get_nowait()
            st.session_state.alerts_display_dequeue.appendleft(alert)
            st.session_state.total_alerts += 1
    except:
        pass
    
    # Display alerts
    if st.session_state.total_alerts > 0:
        # Create a container with fixed height and scrollable content
        with st.container(height=ALERT_HEIGHT):  # Adjust height as needed (in pixels)
            for i, alert in enumerate(st.session_state.alerts_display_dequeue):

                last_alert_timestamp = alert['timestamp']
                st.session_state.last_alert_timestamp = last_alert_timestamp  # save last alert time() UTC

                alert_local_time = (
                    datetime.datetime
                        .fromtimestamp(last_alert_timestamp, tz=datetime.timezone.utc)
                        .astimezone()
                        .strftime('%Y-%m-%d %H:%M:%S')
                )
                
                st.error(f"**Alert:** {alert['alert_msg']}")
                if alert['image'] is not None:  # None when decoding error
                    st.image(
                        alert['image'],
                        use_container_width=True,
                        caption=f"Frame {alert['frame_id']} - {alert_local_time}"
                    )
                
                if i < len(st.session_state.alerts_display_dequeue) - 1:
                    st.divider()
    else:
        st.info("📭 No alerts received yet")


# ================================================================
# Main Application
# ================================================================

# main reruns at every user interaction
def main():
    
    # Configure page
    st.set_page_config(
        page_title="Video Stream & Alerts Monitor",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize services (returns True only on first initialization / main loop)
    initialized = initialize_services(
        ws_host=WEBRTC_HOST,
        ws_port=WEBSOCKET_PORT,
        webrtc_host=WEBRTC_HOST,
        webrtc_port=WEBRTC_PORT,
        webrtc_stream=WEBRTC_STREAM_NAME,
        webrtc_stun=WEBRTC_STUN_SERVER,
    )
    
    # Sidebar configuration
    with st.sidebar:
        st.image(LOGO, width=LOGO_WIDTH)
        
        st.subheader("Video stream configuration")
        st.text_input(
            "WebRTC stream URL",
            value=WEBRTC_URL,
            help="The WebRTC stream URL from media server",
            disabled=True,
        )

        st.text_input(
            "WebRTC stream name",
            value=WEBRTC_STREAM_NAME,
            help="The WebRTC stream name",
            disabled=True,
        )

        st.text_input(
            "WebRTC STUN server",
            value=WEBRTC_STUN_SERVER,
            help="Enter the WebRTC STUN server address",
            disabled=True,
        )
        
        st.subheader("Alerts stream configuration")
        st.text_input(
            "WEBSOCKET server URL",
            value=WEBSOCKET_URL,
            help="The URL of the Websocket server sending alerts",
            disabled=True,
        )
        
        st.divider()

        # Process alerts metrics in a fragment that runs independently
        update_metrics()

        if st.button("Clear displayed alerts", type="secondary"):
            if 'alerts_display_dequeue' in st.session_state:
                st.session_state.alerts_display_dequeue.clear()
                logger.info("Alerts cleared")
    
    # Main content
    left_col, right_col = st.columns([3, 2])
    
    with left_col:
        st.header("🎥 Live Video Stream")
        
        # Only create WebRTC component on startup
        if initialized:
            # Create an HTML component able to receive a WebRTC video stream
            html_video_player = get_video_player(
                webrtc_url=WEBRTC_URL,
                stream_name=WEBRTC_STREAM_NAME,
                stun_server=WEBRTC_STUN_SERVER,
            )
            # Store the HTML in session state
            st.session_state.webrtc_html = html_video_player
            logger.info(f"WebRTC component rendered")

        # Render the component using cached state
        components.html(st.session_state.webrtc_html, height=HTML_HEIGHT)
    
    with right_col:
        st.header("🚨 Alert Feed")
        # Process alerts in a fragment that runs independently
        process_alerts()


if __name__ == "__main__":
    main()

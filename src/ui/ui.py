import streamlit as st
from collections import deque
import queue
import logging
from alert_receiver import TCPAlertReceiver
from video_player import get_video_player
from constants import TCP_PORT, MEDIAMTX_WEBRTC_URL, STREAM_NAME, STUN_SERVER, ALERT_BOX_TIMEDIFF, MAX_ALERT_QUEUE_SIZE, MAX_ALERTS_DISPAYED, REFRESH_STATS, REFRESH_ALERTS, CONTAINER
import streamlit.components.v1 as components
import datetime

# ================================================================
# Logging Configuration
# ================================================================
logfile_name = "ui.log"
log_path = f"/app/logs/{logfile_name}" if CONTAINER else f"../../logs/{logfile_name}"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_path, mode='w')]
)

logger = logging.getLogger("ui")


# ================================================================
# initialization
# ================================================================

assets_path = f"/app/assets" if CONTAINER else f"../../assets"

def initialize_services(tcp_port, webrtc_url, webrtc_stream, webrtc_stun) -> bool:
    """Initialize all services automatically"""
    
    initialized = False

    if 'services_initialized' not in st.session_state:
        
        st.session_state.services_initialized = True
        
        # Initialize session state
        st.session_state.alerts = deque(maxlen=MAX_ALERTS_DISPAYED)
        st.session_state.alert_queue = queue.Queue(maxsize=MAX_ALERT_QUEUE_SIZE)
        st.session_state.total_alerts = 0
        
        # Initialize WebRTC config
        st.session_state.webrtc_config = {'url': webrtc_url,'stream': webrtc_stream,'stun': webrtc_stun}
        logger.info(f"WebRTC config has been setup as: {st.session_state.webrtc_config}")
        
        # Start TCP receiver
        st.session_state.tcp_receiver = TCPAlertReceiver(tcp_port, st.session_state.alert_queue)
        st.session_state.tcp_thread = st.session_state.tcp_receiver.start()
        st.session_state.current_tcp_port = tcp_port
        logger.info(f"TCP service initialized and started on port {tcp_port}")

        initialized = True

    return initialized


# ================================================================
# Alert Processing Fragments
# ================================================================

@st.fragment(run_every=REFRESH_STATS)
def update_metrics():

    if "last_alert_timestamp" not in st.session_state:
        st.session_state.last_alert_timestamp = None

    if st.session_state.last_alert_timestamp is None:
        st.success("No alerts received yet")
    else:
        current_datetime_utc = datetime.datetime.now(datetime.timezone.utc)
        current_datetime_utc = current_datetime_utc.replace(tzinfo=None)
        logger.info(current_datetime_utc)
        logger.info(st.session_state.last_alert_timestamp)

        seconds_passed = int((current_datetime_utc - st.session_state.last_alert_timestamp).total_seconds())
        logger.info(seconds_passed)
        minutes_passed = seconds_passed // 60

        # Convert UTC timestamp to local timezone for display
        last_alert_local = st.session_state.last_alert_timestamp.replace(tzinfo=datetime.timezone.utc).astimezone()
        last_alert_hms = last_alert_local.strftime('%H:%M:%S')
        logger.info(last_alert_hms)

        if seconds_passed < 60:
            text = f"Alert received {seconds_passed} seconds ago ({last_alert_hms})"
        else:
            text = f"Alert received {minutes_passed} minutes ago ({last_alert_hms})"
        
        if seconds_passed > ALERT_BOX_TIMEDIFF:
            st.warning(text)
        else:
            st.error(text)
    
    st.metric("Total Alerts", st.session_state.get('total_alerts', 0))
    st.metric("Displayed Alerts", len(st.session_state.get('alerts', deque())))
        

@st.fragment(run_every=REFRESH_ALERTS)
def process_alerts():
    """Process new alerts in a fragment that runs independently"""
    # Process new alerts
    try:
        while not st.session_state.alert_queue.empty():
            alert = st.session_state.alert_queue.get_nowait()
            st.session_state.alerts.appendleft(alert)
            st.session_state.total_alerts += 1
    except:
        pass
    
    # Display alerts
    if st.session_state.alerts:
        # Create a container with fixed height and scrollable content
        with st.container(height=600):  # Adjust height as needed (in pixels)
            for i, alert in enumerate(st.session_state.alerts):
                
                st.session_state.last_alert_timestamp = alert['timestamp']  # UTC
                alert_local_time = alert['timestamp'].replace(tzinfo=datetime.timezone.utc).astimezone()
                
                st.error(f"**Alert:** {alert['danger']}")
                st.image(
                    alert['image'],
                    use_container_width=True,
                    caption=f"Frame {alert['frame_id']} - {alert_local_time.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                
                if i < len(st.session_state.alerts) - 1:
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
    
    # Sidebar configuration
    with st.sidebar:
        st.image(f"{assets_path}/Logo_Leonardo.png", width=200)
        
        st.subheader("Video stream configuration")
        webrtc_url = st.text_input(
            "WebRTC stream URL",
            value=MEDIAMTX_WEBRTC_URL,
            help="The WebRTC stream URL from MediaMTX media server",
            disabled=True,
        )

        webrtc_stream = st.text_input(
            "WebRTC stream name",
            value=STREAM_NAME,
            help="The WebRTC stream name",
            disabled=True,
        )

        webrtc_stun = st.text_input(
            "WebRTC STUN server",
            value=STUN_SERVER,
            help="Enter the WebRTC STUN server address",
            disabled=True,
        )
        
        st.subheader("Alerts stream configuration")
        tcp_port = st.number_input(
            "TCP Port",
            value=TCP_PORT,
            min_value=1024,
            max_value=65535,
            help="Port for TCP alert connections",
            disabled=True,
        )
        
        st.divider()
        update_metrics()

        if st.button("Clear displayed alerts", type="secondary"):
            if 'alerts' in st.session_state:
                st.session_state.alerts.clear()
                logger.info("Alerts cleared")
    
    # Initialize services (returns True only on initialization)
    initialized = initialize_services(tcp_port, webrtc_url, webrtc_stream, webrtc_stun)
    
    # Main content
    left_col, right_col = st.columns([3, 2])
    
    with left_col:
        st.header("🎥 Live Video Stream")
        
        # Only create WebRTC component on startup
        if initialized:
            # Obtain an HTML component able to receive a video stream
            html_video_player = get_video_player(webrtc_url, webrtc_stream, webrtc_stun)
            # Store the HTML in session state
            st.session_state.webrtc_html = html_video_player
            logger.info(f"WebRTC component rendered")

        # Render the component using cached state
        components.html(st.session_state.webrtc_html, height=600)
    
    with right_col:
        st.header("🚨 Alert Feed")
        
        # Process alerts in a fragment that runs independently
        process_alerts()


if __name__ == "__main__":
    main()
import os
from src.shared.processes.constants import *
from urllib.parse import quote
import re
import math


def fetch_env(key, default):
    """
    Fetches env var;
    If the environmental variable is specified and not empty, returns its value 
    casted to the type of default.
    Otherwise, return the default value.
    If the default value is NOT_SPECIFIED, None is returned.
    """
    value = os.getenv(key)

    # 1. Handle missing or empty environment variable
    if value is None or value.strip() == "":
        return None if default is NOT_SPECIFIED else default

    # 2. If default is NOT_SPECIFIED, we can't infer type, so return as string
    if default is NOT_SPECIFIED:
        return None

    # 3. Cast the value to the type of the default
    
    # Special handling for booleans (since bool("False") is True)
    if isinstance(default, bool):
        return value.lower() in ("true", "1", "yes", "y", "on")
        
    # General casting for int, float, str, etc.
    return type(default)(value)

def is_valid_coord_list(input_string:str, min_couples:int=3):
    # Regex to find patterns like (number, number) of (longitude, latitude)
    # Supports integers, decimals, and negative signs
    coord_pattern = r'\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)'
    
    matches = re.findall(coord_pattern, input_string)
    
    # Check if we found at least the minimum required couples
    if len(matches) < min_couples:
        return False
    
    try:
        for lon_str, lat_str in matches:
            lat, lon = float(lat_str), float(lon_str)
            
            # Latitude must be between -90 and 90
            # Longitude must be between -180 and 180
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                return False
    except ValueError:
        return False

    return True


def preprocess_env_vars():
    """
    Preprocesses environment variables for telemetry listener and database connection.
    Uses fetch_env to get values with appropriate defaults.
    """
    env_vars = {
        # general
        "FPS": fetch_env("FPS", FPS),
        "ALERTS_COOLDOWN_SECONDS": fetch_env("ALERTS_COOLDOWN_SECONDS", ALERTS_COOLDOWN_SECONDS),
        "POISON_PILL_TIMEOUT": fetch_env("POISON_PILL_TIMEOUT", POISON_PILL_TIMEOUT),
        "SHUTDOWN_TIMEOUT": fetch_env("SHUTDOWN_TIMEOUT", SHUTDOWN_TIMEOUT),
        "DB_NAME": fetch_env("DB_NAME", DB_NAME),
        
        # danger detection parameters
        "SAFETY_RADIUS_M": fetch_env("SAFETY_RADIUS_M", SAFETY_RADIUS_M),
        "SLOPE_ANGLE_THRESHOLD": fetch_env("SLOPE_ANGLE_THRESHOLD", SLOPE_ANGLE_THRESHOLD),
        "GEOFENCING_VERTEXES": fetch_env("GEOFENCING_VERTEXES", GEOFENCING_VERTEXES),
        
        # health monitoring parameters
        "SLIDING_WINDOW_SIZE_S": fetch_env("SLIDING_WINDOW_SIZE_S", SLIDING_WINDOW_SIZE_S),

        # drone hardware parameters
        "DRONE_TRUE_FOCAL_LEN_MM": fetch_env("DRONE_TRUE_FOCAL_LEN_MM", DRONE_TRUE_FOCAL_LEN_MM),
        "DRONE_SENSOR_WIDTH_MM": fetch_env("DRONE_SENSOR_WIDTH_MM", DRONE_SENSOR_WIDTH_MM),
        "DRONE_SENSOR_HEIGHT_MM": fetch_env("DRONE_SENSOR_HEIGHT_MM", DRONE_SENSOR_HEIGHT_MM),
        "DRONE_SENSOR_WIDTH_PIXELS": fetch_env("DRONE_SENSOR_WIDTH_PIXELS", DRONE_SENSOR_WIDTH_PIXELS),
        "DRONE_SENSOR_HEIGHT_PIXELS": fetch_env("DRONE_SENSOR_HEIGHT_PIXELS", DRONE_SENSOR_HEIGHT_PIXELS),
        
        # processes queue sizes
        "MAX_SIZE_FRAME_READER_OUT": fetch_env("MAX_SIZE_FRAME_READER_OUT", MAX_SIZE_FRAME_READER_OUT),
        "MAX_SIZE_TELEMETRY_READER_OUT": fetch_env("MAX_SIZE_TELEMETRY_READER_OUT", MAX_SIZE_TELEMETRY_READER_OUT),
        "MAX_SIZE_DETECTION_IN": fetch_env("MAX_SIZE_DETECTION_IN", MAX_SIZE_DETECTION_IN),
        "MAX_SIZE_SEGMENTATION_IN": fetch_env("MAX_SIZE_SEGMENTATION_IN", MAX_SIZE_SEGMENTATION_IN),
        "MAX_SIZE_GEO_IN": fetch_env("MAX_SIZE_GEO_IN", MAX_SIZE_GEO_IN),
        "MAX_SIZE_DETECTION_RESULTS": fetch_env("MAX_SIZE_DETECTION_RESULTS", MAX_SIZE_DETECTION_RESULTS),
        "MAX_SIZE_SEGMENTATION_RESULTS": fetch_env("MAX_SIZE_SEGMENTATION_RESULTS", MAX_SIZE_SEGMENTATION_RESULTS),   
        "MAX_SIZE_GEO_RESULTS": fetch_env("MAX_SIZE_GEO_RESULTS", MAX_SIZE_GEO_RESULTS),
        "MAX_SIZE_MODELS_ALIGNMENT_RESULTS": fetch_env("MAX_SIZE_MODELS_ALIGNMENT_RESULTS", MAX_SIZE_MODELS_ALIGNMENT_RESULTS),
        "MAX_SIZE_DANGER_DETECTION_RESULT": fetch_env("MAX_SIZE_DANGER_DETECTION_RESULT", MAX_SIZE_DANGER_DETECTION_RESULT),
        "MAX_SIZE_VIDEO_STREAM": fetch_env("MAX_SIZE_VIDEO_STREAM", MAX_SIZE_VIDEO_STREAM),
        "MAX_SIZE_NOTIFICATIONS_STREAM": fetch_env("MAX_SIZE_NOTIFICATIONS_STREAM", MAX_SIZE_NOTIFICATIONS_STREAM),
        "MAX_SIZE_VIDEO_STORAGE": fetch_env("MAX_SIZE_VIDEO_STORAGE", MAX_SIZE_VIDEO_STORAGE),
        
        # video reading
        "VIDEO_STREAM_READER_USERNAME": fetch_env("VIDEO_STREAM_READER_USERNAME", VIDEO_STREAM_READER_USERNAME),
        "VIDEO_STREAM_READER_PASSWORD": fetch_env("VIDEO_STREAM_READER_PASSWORD", VIDEO_STREAM_READER_PASSWORD),
        "VIDEO_STREAM_READER_HOST": fetch_env("VIDEO_STREAM_READER_HOST", VIDEO_STREAM_READER_HOST),
        "VIDEO_STREAM_READER_PROTOCOL": fetch_env("VIDEO_STREAM_READER_PROTOCOL", VIDEO_STREAM_READER_PROTOCOL),   
        "VIDEO_STREAM_READER_PORT": fetch_env("VIDEO_STREAM_READER_PORT", VIDEO_STREAM_READER_PORT),
        "VIDEO_STREAM_READER_STREAM_KEY": fetch_env("VIDEO_STREAM_READER_STREAM_KEY", VIDEO_STREAM_READER_STREAM_KEY),
        "VIDEO_STREAM_READER_CONNECTION_OPEN_TIMEOUT_S": fetch_env("VIDEO_STREAM_READER_CONNECTION_OPEN_TIMEOUT_S", VIDEO_STREAM_READER_CONNECTION_OPEN_TIMEOUT_S),
        "VIDEO_STREAM_READER_RECONNECT_DELAY": fetch_env("VIDEO_STREAM_READER_RECONNECT_DELAY", VIDEO_STREAM_READER_RECONNECT_DELAY),
        "VIDEO_STREAM_READER_MAX_CONSECUTIVE_CONNECTION_FAILURES": fetch_env("VIDEO_STREAM_READER_MAX_CONSECUTIVE_CONNECTION_FAILURES", VIDEO_STREAM_READER_MAX_CONSECUTIVE_CONNECTION_FAILURES),
        "VIDEO_STREAM_READER_FRAME_READ_TIMEOUT_S": fetch_env("VIDEO_STREAM_READER_FRAME_READ_TIMEOUT_S", VIDEO_STREAM_READER_FRAME_READ_TIMEOUT_S),
        "VIDEO_STREAM_READER_FRAME_RETRY_DELAY": fetch_env("VIDEO_STREAM_READER_FRAME_RETRY_DELAY", VIDEO_STREAM_READER_FRAME_RETRY_DELAY),
        "VIDEO_STREAM_READER_FRAME_MAX_CONSECUTIVE_FAILURES": fetch_env("VIDEO_STREAM_READER_FRAME_MAX_CONSECUTIVE_FAILURES", VIDEO_STREAM_READER_FRAME_MAX_CONSECUTIVE_FAILURES),
        "VIDEO_STREAM_READER_BUFFER_SIZE": fetch_env("VIDEO_STREAM_READER_BUFFER_SIZE", VIDEO_STREAM_READER_BUFFER_SIZE),
        "VIDEO_STREAM_READER_QUEUE_PUT_TIMEOUT": fetch_env("VIDEO_STREAM_READER_QUEUE_PUT_TIMEOUT", VIDEO_STREAM_READER_QUEUE_PUT_TIMEOUT),
        
        # telemetry listener parameters
        "TELEMETRY_LISTENER_USERNAME": fetch_env("TELEMETRY_LISTENER_USERNAME", TELEMETRY_LISTENER_USERNAME),
        "TELEMETRY_LISTENER_PASSWORD": fetch_env("TELEMETRY_LISTENER_PASSWORD", TELEMETRY_LISTENER_PASSWORD),
        "TELEMETRY_LISTENER_HOST": fetch_env("TELEMETRY_LISTENER_HOST", TELEMETRY_LISTENER_HOST),
        "TELEMETRY_LISTENER_PROTOCOL": fetch_env("TELEMETRY_LISTENER_PROTOCOL", TELEMETRY_LISTENER_PROTOCOL),
        "TELEMETRY_LISTENER_PORT": fetch_env("TELEMETRY_LISTENER_PORT", TELEMETRY_LISTENER_PORT),
        "TELEMETRY_LISTENER_QOS_LEVEL": fetch_env("TELEMETRY_LISTENER_QOS_LEVEL", TELEMETRY_LISTENER_QOS_LEVEL),
        "TELEMETRY_LISTENER_RECONNECT_DELAY": fetch_env("TELEMETRY_LISTENER_RECONNECT_DELAY", TELEMETRY_LISTENER_RECONNECT_DELAY),   
        "TELEMETRY_LISTENER_MSG_WAIT_TIMEOUT": fetch_env("TELEMETRY_LISTENER_MSG_WAIT_TIMEOUT", TELEMETRY_LISTENER_MSG_WAIT_TIMEOUT),
        "TELEMETRY_LISTENER_MAX_INCOMING_MESSAGES": fetch_env("TELEMETRY_LISTENER_MAX_INCOMING_MESSAGES", TELEMETRY_LISTENER_MAX_INCOMING_MESSAGES),

        # frame telemetry combiner
        "FRAMETELCOMB_MAX_TIME_DIFF": fetch_env("FRAMETELCOMB_MAX_TIME_DIFF", FRAMETELCOMB_MAX_TIME_DIFF),
        "FRAMETELCOMB_QUEUE_GET_TIMEOUT": fetch_env("FRAMETELCOMB_QUEUE_GET_TIMEOUT", FRAMETELCOMB_QUEUE_GET_TIMEOUT),
        "FRAMETELCOMB_QUEUE_PUT_MAX_RETRIES": fetch_env("FRAMETELCOMB_QUEUE_PUT_MAX_RETRIES", FRAMETELCOMB_QUEUE_PUT_MAX_RETRIES),
        "FRAMETELCOMB_QUEUE_PUT_BACKOFF": fetch_env("FRAMETELCOMB_QUEUE_PUT_BACKOFF", FRAMETELCOMB_QUEUE_PUT_BACKOFF),

        # models and annotations
        "MODELS_QUEUE_GET_TIMEOUT": fetch_env("MODELS_QUEUE_GET_TIMEOUT", MODELS_QUEUE_GET_TIMEOUT),
        "MODELS_QUEUE_PUT_TIMEOUT": fetch_env("MODELS_QUEUE_PUT_TIMEOUT", MODELS_QUEUE_PUT_TIMEOUT),
        
        "ANNOTATION_QUEUE_GET_TIMEOUT": fetch_env("ANNOTATION_QUEUE_GET_TIMEOUT", ANNOTATION_QUEUE_GET_TIMEOUT),
        "ANNOTATION_QUEUE_PUT_TIMEOUT": fetch_env("ANNOTATION_QUEUE_PUT_TIMEOUT", ANNOTATION_QUEUE_PUT_TIMEOUT),
        "ANNOTATION_MAX_PUT_ALERT_CONSECUTIVE_FAILURES": fetch_env("ANNOTATION_MAX_PUT_ALERT_CONSECUTIVE_FAILURES", ANNOTATION_MAX_PUT_ALERT_CONSECUTIVE_FAILURES),
        "ANNOTATION_MAX_PUT_VIDEO_CONSECUTIVE_FAILURES": fetch_env("ANNOTATION_MAX_PUT_VIDEO_CONSECUTIVE_FAILURES", ANNOTATION_MAX_PUT_VIDEO_CONSECUTIVE_FAILURES),

        # alerts writer
        
        "ALERTS_QUEUE_GET_TIMEOUT": fetch_env("ALERTS_QUEUE_GET_TIMEOUT", ALERTS_QUEUE_GET_TIMEOUT),
        "ALERTS_MAX_CONSECUTIVE_FAILURES": fetch_env("ALERTS_MAX_CONSECUTIVE_FAILURES", ALERTS_MAX_CONSECUTIVE_FAILURES),
        "ALERTS_JPEG_COMPRESSION_QUALITY": fetch_env("ALERTS_JPEG_COMPRESSION_QUALITY", ALERTS_JPEG_COMPRESSION_QUALITY),
        
        "WS_MANAGER_BROADCAST_TIMEOUT": fetch_env("WS_MANAGER_BROADCAST_TIMEOUT", WS_MANAGER_BROADCAST_TIMEOUT),
        "WS_MANAGER_PING_INTERVAL": fetch_env("WS_MANAGER_PING_INTERVAL", WS_MANAGER_PING_INTERVAL),
        "WS_MANAGER_PING_TIMEOUT": fetch_env("WS_MANAGER_PING_TIMEOUT", WS_MANAGER_PING_TIMEOUT),
        "WS_MANAGER_THREAD_CLOSE_TIMEOUT": fetch_env("WS_MANAGER_THREAD_CLOSE_TIMEOUT", WS_MANAGER_THREAD_CLOSE_TIMEOUT),

        "DB_USERNAME": fetch_env("DB_USERNAME", DB_USERNAME),
        "DB_PASSWORD": fetch_env("DB_PASSWORD", DB_PASSWORD),
        "DB_HOST": fetch_env("DB_HOST", DB_HOST),
        "DB_SERVICE": fetch_env("DB_SERVICE", DB_SERVICE),
        "DB_PORT": fetch_env("DB_PORT", DB_PORT),
        "DB_MANAGER_QUEUE_SIZE": fetch_env("DB_MANAGER_QUEUE_SIZE", DB_MANAGER_QUEUE_SIZE),
        "DB_MANAGER_POOL_SIZE": fetch_env("DB_MANAGER_POOL_SIZE", DB_MANAGER_POOL_SIZE),
        "DB_MANAGER_MAX_OVERFLOW": fetch_env("DB_MANAGER_MAX_OVERFLOW", DB_MANAGER_MAX_OVERFLOW),
        "DB_MANAGER_QUEUE_WAIT_TIMEOUT": fetch_env("DB_MANAGER_QUEUE_WAIT_TIMEOUT", DB_MANAGER_QUEUE_WAIT_TIMEOUT),
        "DB_MANAGER_THREAD_CLOSE_TIMEOUT": fetch_env("DB_MANAGER_THREAD_CLOSE_TIMEOUT", DB_MANAGER_THREAD_CLOSE_TIMEOUT),

        # out video writer
        "VIDEO_WRITER_GET_FRAME_TIMEOUT": fetch_env("VIDEO_WRITER_GET_FRAME_TIMEOUT", VIDEO_WRITER_GET_FRAME_TIMEOUT),
        "VIDEO_WRITER_HANDOFF_TIMEOUT": fetch_env("VIDEO_WRITER_HANDOFF_TIMEOUT", VIDEO_WRITER_HANDOFF_TIMEOUT),
        
        "VIDEO_OUT_STREAM_USERNAME": fetch_env("VIDEO_OUT_STREAM_USERNAME", VIDEO_OUT_STREAM_USERNAME),
        "VIDEO_OUT_STREAM_PASSWORD": fetch_env("VIDEO_OUT_STREAM_PASSWORD", VIDEO_OUT_STREAM_PASSWORD),
        "VIDEO_OUT_STREAM_PROTOCOL": fetch_env("VIDEO_OUT_STREAM_PROTOCOL", VIDEO_OUT_STREAM_PROTOCOL),
        "VIDEO_OUT_STREAM_PORT": fetch_env("VIDEO_OUT_STREAM_PORT", VIDEO_OUT_STREAM_PORT),
        "VIDEO_OUT_STREAM_QUEUE_GET_TIMEOUT": fetch_env("VIDEO_OUT_STREAM_QUEUE_GET_TIMEOUT", VIDEO_OUT_STREAM_QUEUE_GET_TIMEOUT),
        "VIDEO_OUT_STREAM_FFMPEG_STARTUP_TIMEOUT": fetch_env("VIDEO_OUT_STREAM_FFMPEG_STARTUP_TIMEOUT", VIDEO_OUT_STREAM_FFMPEG_STARTUP_TIMEOUT),
        "VIDEO_OUT_STREAM_FFMPEG_SHUTDOWN_TIMEOUT": fetch_env("VIDEO_OUT_STREAM_FFMPEG_SHUTDOWN_TIMEOUT", VIDEO_OUT_STREAM_FFMPEG_SHUTDOWN_TIMEOUT),
        "VIDEO_OUT_STREAM_STARTUP_TIMEOUT": fetch_env("VIDEO_OUT_STREAM_STARTUP_TIMEOUT", VIDEO_OUT_STREAM_STARTUP_TIMEOUT),   
        "VIDEO_OUT_STREAM_SHUTDOWN_TIMEOUT": fetch_env("VIDEO_OUT_STREAM_SHUTDOWN_TIMEOUT", VIDEO_OUT_STREAM_SHUTDOWN_TIMEOUT),
        
        "VIDEO_OUT_STORE_SERVICE": fetch_env("VIDEO_OUT_STORE_SERVICE", VIDEO_OUT_STORE_SERVICE),
        "VIDEO_OUT_STORE_PORT": fetch_env("VIDEO_OUT_STORE_PORT", VIDEO_OUT_STORE_PORT),
        "VIDEO_OUT_STORE_DELETE_LOCAL_ON_SUCCESS": fetch_env("VIDEO_OUT_STORE_DELETE_LOCAL_ON_SUCCESS", VIDEO_OUT_STORE_DELETE_LOCAL_ON_SUCCESS),
        "VIDEO_OUT_STORE_QUEUE_GET_TIMEOUT": fetch_env("VIDEO_OUT_STORE_QUEUE_GET_TIMEOUT", VIDEO_OUT_STORE_QUEUE_GET_TIMEOUT),
        "VIDEO_OUT_STORE_MAX_UPLOAD_RETRIES": fetch_env("VIDEO_OUT_STORE_MAX_UPLOAD_RETRIES", VIDEO_OUT_STORE_MAX_UPLOAD_RETRIES),
        "VIDEO_OUT_STORE_RETRY_BACKOFF_TIME": fetch_env("VIDEO_OUT_STORE_RETRY_BACKOFF_TIME", VIDEO_OUT_STORE_RETRY_BACKOFF_TIME),
    }

    # preprocessing to ensure correctness:
    
    # fps must be a positive number
    if not env_vars["FPS"] > 0:
        raise ValueError("FPS environment variable must be a positive integer.")
    
    # alerts cooldwon must be a positive float
    if not env_vars["ALERTS_COOLDOWN_SECONDS"] > 0:
        raise ValueError("ALERTS_COOLDOWN_SECONDS environment variable must be a positive float.")

    # poison pil ltimeout must be a positive float
    if not env_vars["POISON_PILL_TIMEOUT"] > 0:
        raise ValueError("POISON_PILL_TIMEOUT environment variable must be a positive float.")

    # shutdown timeout must be a positive float
    if not env_vars["SHUTDOWN_TIMEOUT"] > 0:
        raise ValueError("SHUTDOWN_TIMEOUT environment variable must be a positive float.")
    
    # ------------------ DANGER DETECTION ---------------

    # safety radius must be a positive float
    if not env_vars["SAFETY_RADIUS_M"] > 0:
        raise ValueError("SAFETY_RADIUS_M environment variable must be a positive float.")
    
    # slope angle threshold must be between 0 and 90 degrees
    if not (0 <= env_vars["SLOPE_ANGLE_THRESHOLD"] <= 90):
        raise ValueError("SLOPE_ANGLE_THRESHOLD environment variable must be between 0 and 90 degrees.")
    

    # GEOFENCING_VERTEXES is either None or a list of (longitude, latitude) couples
    if env_vars["GEOFENCING_VERTEXES"] is not None and not is_valid_coord_list(env_vars["GEOFENCING_VERTEXES"]):
        raise ValueError("If specified, GEOFENCING_VERTEXES environment variable must be a string like (long1, lat1), (long2, lat2), (long3, lat3), ... . At least 3 points are expected")


    # ---------------------- HEALTH MONIOTIRNG ------------

    # sliding window size must be a positive float
    if not env_vars["SLIDING_WINDOW_SIZE_S"] > 0:
        raise ValueError("SLIDING_WINDOW_SIZE_S environment variable must be a positive float.")
    
    # ---------------------- DRONE ----------------

    # drone true focal length must be a positive float
    if not env_vars["DRONE_TRUE_FOCAL_LEN_MM"] > 0:
        raise ValueError("DRONE_TRUE_FOCAL_LEN_MM environment variable must be a positive float.")
    
    # drone sensor width must be a positive float
    if not env_vars["DRONE_SENSOR_WIDTH_MM"] > 0:
        raise ValueError("DRONE_SENSOR_WIDTH_MM environment variable must be a positive float.")
    
    # drone sensor height must be a positive float
    if not env_vars["DRONE_SENSOR_HEIGHT_MM"] > 0:
        raise ValueError("DRONE_SENSOR_HEIGHT_MM environment variable must be a positive float.")
    
    # drone sensor width in pixels must be a positive integer
    if not env_vars["DRONE_SENSOR_WIDTH_PIXELS"] > 0:
        raise ValueError("DRONE_SENSOR_WIDTH_PIXELS environment variable must be a positive integer.")
    
    # drone sensor height in pixels must be a positive integer
    if not env_vars["DRONE_SENSOR_HEIGHT_PIXELS"] > 0:
        raise ValueError("DRONE_SENSOR_HEIGHT_PIXELS environment variable must be a positive integer.")
    
    # sensor_width_pixels/sensor_height_pixels MUST be equal to sensor_width_mm/sensor_height_mm!!!!
    physical_ratio = env_vars["DRONE_SENSOR_WIDTH_MM"] / env_vars["DRONE_SENSOR_HEIGHT_MM"]
    pixel_ratio = env_vars["DRONE_SENSOR_WIDTH_PIXELS"] / env_vars["DRONE_SENSOR_HEIGHT_PIXELS"]
    if not math.isclose(physical_ratio, pixel_ratio, rel_tol=1e-5):
        raise ValueError(
            f"Drone camera sensor aspect ratio mismatch! "
            f"Physical ({physical_ratio:.4f}) does not match "
            f"Pixel ({pixel_ratio:.4f}). "
            "Check your sensor dimensions."
        )
    # -------------------------- QUEUES -------------------

    # queue sizes must be non-negative integers
    processes_queue_sizes = [
        "MAX_SIZE_FRAME_READER_OUT",
        "MAX_SIZE_TELEMETRY_READER_OUT",
        "MAX_SIZE_DETECTION_IN",
        "MAX_SIZE_SEGMENTATION_IN",
        "MAX_SIZE_GEO_IN",
        "MAX_SIZE_DETECTION_RESULTS",
        "MAX_SIZE_SEGMENTATION_RESULTS",
        "MAX_SIZE_GEO_RESULTS",
        "MAX_SIZE_MODELS_ALIGNMENT_RESULTS",
        "MAX_SIZE_DANGER_DETECTION_RESULT",
        "MAX_SIZE_VIDEO_STREAM",
        "MAX_SIZE_NOTIFICATIONS_STREAM",
        "MAX_SIZE_VIDEO_STORAGE",
    ]
    for queue_size_key in processes_queue_sizes:
        if not env_vars[queue_size_key] >= 0:
            raise ValueError(f"{queue_size_key} environment variable must be a non-negative integer.")
        
    # ------------------------ VIDEO READING ----------
    
    # video stream reader protocol is allowed value
    env_vars["VIDEO_STREAM_READER_PROTOCOL"] = env_vars["VIDEO_STREAM_READER_PROTOCOL"].lower()
    if env_vars["VIDEO_STREAM_READER_PROTOCOL"] not in VIDEO_STREAM_READER_ALLOWED_PROTOCOLS:
        raise ValueError(f"VIDEO_STREAM_READER_PROTOCOL environment variable must be one of {VIDEO_STREAM_READER_ALLOWED_PROTOCOLS}.")

    # video stream reader port must be a positive integer
    if not env_vars["VIDEO_STREAM_READER_PORT"] > 0:
        raise ValueError("VIDEO_STREAM_READER_PORT environment variable must be a positive integer.")
    
    # escape credentials
    if env_vars["VIDEO_STREAM_READER_USERNAME"] is not None:
        env_vars["VIDEO_STREAM_READER_USERNAME"] = quote(env_vars["VIDEO_STREAM_READER_USERNAME"])
    if env_vars["VIDEO_STREAM_READER_PASSWORD"] is not None:
        env_vars["VIDEO_STREAM_READER_PASSWORD"] = quote(env_vars["VIDEO_STREAM_READER_PASSWORD"])
    

    # CREATE URL
    env_vars["VIDEO_STREAM_READER_URL"] = None
    protocol = f"{env_vars["VIDEO_STREAM_READER_PROTOCOL"]}://"
    host_port_key = f"{env_vars["VIDEO_STREAM_READER_HOST"]}:{env_vars["VIDEO_STREAM_READER_PORT"]}/{env_vars["VIDEO_STREAM_READER_STREAM_KEY"]}"
    
    if (env_vars["VIDEO_STREAM_READER_PROTOCOL"] in [RTMPS, RTSPS]):
        # username and password must be provided for secure protocols
        if (env_vars["VIDEO_STREAM_READER_USERNAME"] is None or env_vars["VIDEO_STREAM_READER_PASSWORD"] is None):
            raise ValueError(f"{env_vars["VIDEO_STREAM_READER_PROTOCOL"]} requires both username and password to be specified")
        else:
            auth_prefix = f"{env_vars["VIDEO_STREAM_READER_USERNAME"]}:{env_vars["VIDEO_STREAM_READER_PASSWORD"]}@"
            env_vars["VIDEO_STREAM_READER_URL"] = f"{protocol}{auth_prefix}{host_port_key}"
    # and can be ignored otheriwse
    else:
        env_vars["VIDEO_STREAM_READER_USERNAME"] = None 
        env_vars["VIDEO_STREAM_READER_PASSWORD"] = None
        env_vars["VIDEO_STREAM_READER_URL"] = f"{protocol}{host_port_key}"
        

    # connection open timeout must be a positive float
    if not env_vars["VIDEO_STREAM_READER_CONNECTION_OPEN_TIMEOUT_S"] > 0:
        raise ValueError("VIDEO_STREAM_READER_CONNECTION_OPEN_TIMEOUT_S environment variable must be a positive float.")
    
    # reconnect delay must be a positive float
    if not env_vars["VIDEO_STREAM_READER_RECONNECT_DELAY"] > 0:
        raise ValueError("VIDEO_STREAM_READER_RECONNECT_DELAY environment variable must be a positive float.")
    
    # max consecutive connection failures must be a non-negative integer
    if not env_vars["VIDEO_STREAM_READER_MAX_CONSECUTIVE_CONNECTION_FAILURES"] >= 0:
        raise ValueError("VIDEO_STREAM_READER_MAX_CONSECUTIVE_CONNECTION_FAILURES environment variable must be a non-negative integer.")
    
    # frame read timeout must be a positive float
    if not env_vars["VIDEO_STREAM_READER_FRAME_READ_TIMEOUT_S"] > 0:
        raise ValueError("VIDEO_STREAM_READER_FRAME_READ_TIMEOUT_S environment variable must be a positive float.")
    
    # frame retry delay must be a positive float
    if not env_vars["VIDEO_STREAM_READER_FRAME_RETRY_DELAY"] > 0:
        raise ValueError("VIDEO_STREAM_READER_FRAME_RETRY_DELAY environment variable must be a positive float.")
    
    # frame max consecutive failures must be a non-negative integer
    if not env_vars["VIDEO_STREAM_READER_FRAME_MAX_CONSECUTIVE_FAILURES"] >= 0:
        raise ValueError("VIDEO_STREAM_READER_FRAME_MAX_CONSECUTIVE_FAILURES environment variable must be a non-negative integer.")
    
    # buffer size must be a positive integer
    if not env_vars["VIDEO_STREAM_READER_BUFFER_SIZE"] > 0:
        raise ValueError("VIDEO_STREAM_READER_BUFFER_SIZE environment variable must be a positive integer.")
    
    # queue put timeout must be a positive float
    if not env_vars["VIDEO_STREAM_READER_QUEUE_PUT_TIMEOUT"] > 0:
        raise ValueError("VIDEO_STREAM_READER_QUEUE_PUT_TIMEOUT environment variable must be a positive float.")
    
    # -------------------------- TELEMETRY READER --------------------------

    # telemetry listener protocol is allowed value
    env_vars["TELEMETRY_LISTENER_PROTOCOL"] = env_vars["TELEMETRY_LISTENER_PROTOCOL"].lower()
    if env_vars["TELEMETRY_LISTENER_PROTOCOL"] not in TELEMETRY_LISTENER_ALLOWED_PROTOCOLS:
        raise ValueError(f"TELEMETRY_LISTENER_PROTOCOL environment variable must be one of {TELEMETRY_LISTENER_ALLOWED_PROTOCOLS}.")
    
    # telemetry listener port must be a positive integer
    if not env_vars["TELEMETRY_LISTENER_PORT"] > 0:
        raise ValueError("TELEMETRY_LISTENER_PORT environment variable must be a positive integer.")
    
    # escape credentials
    if env_vars["TELEMETRY_LISTENER_USERNAME"] is not None:
        env_vars["TELEMETRY_LISTENER_USERNAME"] = quote(env_vars["TELEMETRY_LISTENER_USERNAME"])
    if env_vars["TELEMETRY_LISTENER_PASSWORD"] is not None:
        env_vars["TELEMETRY_LISTENER_PASSWORD"] = quote(env_vars["TELEMETRY_LISTENER_PASSWORD"])
    
    if (env_vars["TELEMETRY_LISTENER_PROTOCOL"] == MQTTS and (
        env_vars["TELEMETRY_LISTENER_USERNAME"] is None or 
        env_vars["TELEMETRY_LISTENER_PASSWORD"] is None)):
        raise ValueError(f"{env_vars["TELEMETRY_LISTENER_PROTOCOL"]} requires both username and password to be specified")
    elif env_vars["TELEMETRY_LISTENER_PROTOCOL"] == MQTT:
        env_vars["TELEMETRY_LISTENER_USERNAME"] = None 
        env_vars["TELEMETRY_LISTENER_PASSWORD"] = None


    # qos level must be 0, 1, or 2
    if env_vars["TELEMETRY_LISTENER_QOS_LEVEL"] not in (0, 1, 2):
        raise ValueError("TELEMETRY_LISTENER_QOS_LEVEL environment variable must be 0, 1, or 2.")
    
    # reconnect delay must be a positive float
    if not env_vars["TELEMETRY_LISTENER_RECONNECT_DELAY"] > 0:
        raise ValueError("TELEMETRY_LISTENER_RECONNECT_DELAY environment variable must be a positive float.")
    
    # message wait timeout must be a positive float
    if not env_vars["TELEMETRY_LISTENER_MSG_WAIT_TIMEOUT"] > 0:
        raise ValueError("TELEMETRY_LISTENER_MSG_WAIT_TIMEOUT environment variable must be a positive float.")
    
    # max incoming messages must be a positive integer
    if not env_vars["TELEMETRY_LISTENER_MAX_INCOMING_MESSAGES"] > 0:
        raise ValueError("TELEMETRY_LISTENER_MAX_INCOMING_MESSAGES environment variable must be a positive integer.")
    
    # -------------------------- FRAME + TELEMETRY COMBINING ------------

    # max time diff must be a non-negative float
    if not env_vars["FRAMETELCOMB_MAX_TIME_DIFF"] >= 0:
        raise ValueError("FRAMETELCOMB_MAX_TIME_DIFF environment variable must be a non-negative float.")
    
    # queue get timeout must be a positive float
    if not env_vars["FRAMETELCOMB_QUEUE_GET_TIMEOUT"] > 0:
        raise ValueError("FRAMETELCOMB_QUEUE_GET_TIMEOUT environment variable must be a positive float.")
    
    # queue put max retries must be a non-negative integer
    if not env_vars["FRAMETELCOMB_QUEUE_PUT_MAX_RETRIES"] >= 0:
        raise ValueError("FRAMETELCOMB_QUEUE_PUT_MAX_RETRIES environment variable must be a non-negative integer.")
    
    # queue put backoff must be a positive float
    if not env_vars["FRAMETELCOMB_QUEUE_PUT_BACKOFF"] > 0:
        raise ValueError("FRAMETELCOMB_QUEUE_PUT_BACKOFF environment variable must be a positive float.")
    
    # -------------------------- MODELS & ANNOTATIONS -------------------

    # models queue get timeout must be a positive float
    if not env_vars["MODELS_QUEUE_GET_TIMEOUT"] > 0:
        raise ValueError("MODELS_QUEUE_GET_TIMEOUT environment variable must be a positive float.")
    
    # models queue put timeout must be a positive float
    if not env_vars["MODELS_QUEUE_PUT_TIMEOUT"] > 0:
        raise ValueError("MODELS_QUEUE_PUT_TIMEOUT environment variable must be a positive float.")

    # annotation queue get timeout must be a positive float
    if not env_vars["ANNOTATION_QUEUE_GET_TIMEOUT"] > 0:
        raise ValueError("ANNOTATION_QUEUE_GET_TIMEOUT environment variable must be a positive float.")
    
    # annotation queue put timeout must be a positive float
    if not env_vars["ANNOTATION_QUEUE_PUT_TIMEOUT"] > 0:
        raise ValueError("ANNOTATION_QUEUE_PUT_TIMEOUT environment variable must be a positive float.")
    
    # max put alert consecutive failures must be a non-negative integer
    if not env_vars["ANNOTATION_MAX_PUT_ALERT_CONSECUTIVE_FAILURES"] >= 0:
        raise ValueError("ANNOTATION_MAX_PUT_ALERT_CONSECUTIVE_FAILURES environment variable must be a non-negative integer.")
    
    # max put video consecutive failures must be a non-negative integer
    if not env_vars["ANNOTATION_MAX_PUT_VIDEO_CONSECUTIVE_FAILURES"] >= 0:
        raise ValueError("ANNOTATION_MAX_PUT_VIDEO_CONSECUTIVE_FAILURES environment variable must be a non-negative integer.")

    # -------------------------- ALERTS WRITER ---------------------------

    # alerts queue get timeout must be a positive float
    if not env_vars["ALERTS_QUEUE_GET_TIMEOUT"] > 0:
        raise ValueError("ALERTS_QUEUE_GET_TIMEOUT environment variable must be a positive float.")
    
    # alerts max consecutive failures must be a non-negative integer
    if not env_vars["ALERTS_MAX_CONSECUTIVE_FAILURES"] >= 0:
        raise ValueError("ALERTS_MAX_CONSECUTIVE_FAILURES environment variable must be a non-negative integer.")
    
    # alerts jpeg compression quality must be between 0 and 100
    if not (0 < env_vars["ALERTS_JPEG_COMPRESSION_QUALITY"] <= 100):
        raise ValueError("ALERTS_JPEG_COMPRESSION_QUALITY environment variable must be between 0 (excluded) and 100.")
    
    # websocket manager broadcast timeout must be a positive float
    if not env_vars["WS_MANAGER_BROADCAST_TIMEOUT"] > 0:
        raise ValueError("WS_MANAGER_BROADCAST_TIMEOUT environment variable must be a positive float.")
    
    # websocket manager ping interval must be a positive float
    if not env_vars["WS_MANAGER_PING_INTERVAL"] > 0:
        raise ValueError("WS_MANAGER_PING_INTERVAL environment variable must be a positive float.")
    
    # websocket manager ping timeout must be a positive float
    if not env_vars["WS_MANAGER_PING_TIMEOUT"] > 0:
        raise ValueError("WS_MANAGER_PING_TIMEOUT environment variable must be a positive float.")
    
    # websocket manager thread close timeout must be a positive float
    if not env_vars["WS_MANAGER_THREAD_CLOSE_TIMEOUT"] > 0:
        raise ValueError("WS_MANAGER_THREAD_CLOSE_TIMEOUT environment variable must be a positive float.")
    
    # -------------------------- DATABASE CONNECTION --------------------

    # db service must be an allowed value
    env_vars["DB_SERVICE"] = env_vars["DB_SERVICE"].lower() if env_vars["DB_SERVICE"] is not None else None
    if env_vars["DB_SERVICE"] not in DB_ALLOWED_SERVICES:
        raise ValueError(f"DB_SERVICE environment variable must be one of {DB_ALLOWED_SERVICES}.")
    
    # escape credentials
    if env_vars["DB_USERNAME"] is not None:
        env_vars["DB_USERNAME"] = quote(env_vars["DB_USERNAME"])
    if env_vars["DB_PASSWORD"] is not None:
        env_vars["DB_PASSWORD"] = quote(env_vars["DB_PASSWORD"])

    # db port must be a positive integer
    if not env_vars["DB_PORT"] > 0:
        raise ValueError("DB_PORT environment variable must be a positive integer.")

    # db manager queue size must be a positive integer
    if not env_vars["DB_MANAGER_QUEUE_SIZE"] > 0:
        raise ValueError("DB_MANAGER_QUEUE_SIZE environment variable must be a positive integer.")

    # db ,amager pool size must be a positive integer
    if not env_vars["DB_MANAGER_POOL_SIZE"] > 0:
        raise ValueError("DB_MANAGER_POOL_SIZE environment variable must be a positive integer.")

    # db manager max overflow must be a non-negative integer
    if not env_vars["DB_MANAGER_MAX_OVERFLOW"] >= 0:
        raise ValueError("DB_MANAGER_MAX_OVERFLOW environment variable must be a non-negative integer.")
    
    # db manager queue wait timeout must be a positive float
    if not env_vars["DB_MANAGER_QUEUE_WAIT_TIMEOUT"] > 0:
        raise ValueError("DB_MANAGER_QUEUE_WAIT_TIMEOUT environment variable must be a positive float.")
    
    # db manager thread close timeout must be a positive float
    if not env_vars["DB_MANAGER_THREAD_CLOSE_TIMEOUT"] > 0:
        raise ValueError("DB_MANAGER_THREAD_CLOSE_TIMEOUT environment variable must be a positive float.")
    
    # ----------------- OUT VIDEO WRITER --------------------

    # video writer get timeout must be a positive float
    if not env_vars["VIDEO_WRITER_GET_FRAME_TIMEOUT"] > 0:
        raise ValueError("VIDEO_WRITER_GET_FRAME_TIMEOUT environment variable must be a positive float.")
    
    # video writer handoff timeout must be a positive float
    if not env_vars["VIDEO_WRITER_HANDOFF_TIMEOUT"] > 0:
        raise ValueError("VIDEO_WRITER_HANDOFF_TIMEOUT environment variable must be a positive float.")
    

    # video stream reader protocol is allowed value
    env_vars["VIDEO_OUT_STREAM_PROTOCOL"] = env_vars["VIDEO_OUT_STREAM_PROTOCOL"].lower()
    if env_vars["VIDEO_OUT_STREAM_PROTOCOL"] not in VIDEO_OUT_STREAM_ALLOWED_PROTOCOLS:
        raise ValueError(f"VIDEO_OUT_STREAM_PROTOCOL environment variable must be one of {VIDEO_OUT_STREAM_ALLOWED_PROTOCOLS}.")
    
    # video stream reader port must be a positive integer
    if not env_vars["VIDEO_OUT_STREAM_PORT"] > 0:
        raise ValueError("VIDEO_OUT_STREAM_PORT environment variable must be a positive integer.")
    
    # escape credentials
    if env_vars["VIDEO_OUT_STREAM_USERNAME"] is not None:
        env_vars["VIDEO_OUT_STREAM_USERNAME"] = quote(env_vars["VIDEO_OUT_STREAM_USERNAME"])
    if env_vars["VIDEO_OUT_STREAM_PASSWORD"] is not None:
        env_vars["VIDEO_OUT_STREAM_PASSWORD"] = quote(env_vars["VIDEO_OUT_STREAM_PASSWORD"])
    
    # CREATE URL
    env_vars["VIDEO_OUT_STREAM_URL"] = None
    protocol = f"{env_vars["VIDEO_OUT_STREAM_PROTOCOL"]}://"
    host_port_key = f"{env_vars["VIDEO_OUT_STREAM_HOST"]}:{env_vars["VIDEO_OUT_STREAM_PORT"]}/{env_vars["VIDEO_OUT_STREAM_STREAM_KEY"]}"
    
    if (env_vars["VIDEO_OUT_STREAM_PROTOCOL"] in [RTMPS]):
        # username and password must be provided for secure protocols
        if (env_vars["VIDEO_OUT_STREAM_USERNAME"] is None or env_vars["VIDEO_OUT_STREAM_PASSWORD"] is None):
            raise ValueError(f"{env_vars["VIDEO_OUT_STREAM_PROTOCOL"]} requires both username and password to be specified")
        else:
            auth_prefix = f"{env_vars["VIDEO_OUT_STREAM_USERNAME"]}:{env_vars["VIDEO_OUT_STREAM_PASSWORD"]}@"
            env_vars["VIDEO_OUT_STREAM_URL"] = f"{protocol}{auth_prefix}{host_port_key}"
    # and can be ignored otheriwse
    else:
        env_vars["VIDEO_OUT_STREAM_USERNAME"] = None 
        env_vars["VIDEO_OUT_STREAM_PASSWORD"] = None
        env_vars["VIDEO_OUT_STREAM_URL"] = f"{protocol}{host_port_key}"
    
    # Timeouts must be positive
    stream_timeouts = [
        "VIDEO_OUT_STREAM_QUEUE_GET_TIMEOUT",
        "VIDEO_OUT_STREAM_FFMPEG_STARTUP_TIMEOUT",
        "VIDEO_OUT_STREAM_FFMPEG_SHUTDOWN_TIMEOUT",
        "VIDEO_OUT_STREAM_STARTUP_TIMEOUT",
        "VIDEO_OUT_STREAM_SHUTDOWN_TIMEOUT"
    ]
    for var in stream_timeouts:
        if not env_vars[var] > 0:
            raise ValueError(f"{var} environment variable must be a positive number.")

    # --- VIDEO_OUT_STORE  ---

    # Services must be in the allowed list
    env_vars["VIDEO_OUT_STORE_SERVICE"] = env_vars["VIDEO_OUT_STORE_SERVICE"].lower()
    if env_vars["VIDEO_OUT_STORE_SERVICE"] not in VIDEO_OUT_STORE_ALLOWED_SERVICES:
        raise ValueError(f"VIDEO_OUT_STORE_SERVICE must be one of {VIDEO_OUT_STORE_ALLOWED_SERVICES}.")

    # Store port must be a positive integer
    if not env_vars["VIDEO_OUT_STORE_PORT"] > 0:
        raise ValueError("VIDEO_OUT_STORE_PORT must be a positive integer.")

    # Queue timeout must be a positive float
    if not env_vars["VIDEO_OUT_STORE_QUEUE_GET_TIMEOUT"] > 0:
        raise ValueError("VIDEO_OUT_STORE_QUEUE_GET_TIMEOUT must be a positive float.")

    # Retries must be a non-negative integer
    if not env_vars["VIDEO_OUT_STORE_MAX_UPLOAD_RETRIES"] >= 0:
        raise ValueError("VIDEO_OUT_STORE_MAX_UPLOAD_RETRIES must be a non-negative integer.")

    # Backoff time must be a positive number
    if not env_vars["VIDEO_OUT_STORE_RETRY_BACKOFF_TIME"] > 0:
        raise ValueError("VIDEO_OUT_STORE_RETRY_BACKOFF_TIME must be a positive float.")
    
    # TODO: impleemt service specific URL and CLASSES for STORAGE
    env_vars["VIDEO_OUT_STORE_URL"] = None

    
    return env_vars
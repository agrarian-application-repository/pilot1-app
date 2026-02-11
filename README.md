![AGRARIAN](assets/agrarian.png)

# Node type
Cloud / GPU

# Usage Instructions

## Expected inputs
- A RTMP video stream, with key `drone` (configurable), to be received on port 1935
- A stream of MQTT(S) telemetry data, on port(s) 1883/8883,  topics: 
    * telemetry/latitude
    * telemetry/longitude
    * telemetry/rel_alt
    * telemetry/gb_yaw
- optinal: mount a `dem.tif` and/or `dem_mask.tif` files into container `DEM/` folder to unlock steep slope mapping capabilities
    
## Outputs
Client UI connects to:
- mediaserver: UI fetchs `annot` (configurable) stream from mediaserver, port 8889 (WebRTC)
- app: UI receives push alerts from app-internal websocket, port 8443 (WEBSOCKET)
- db: app pushes alerts to the AGRARIAN DB
- [ final recording: pushed to a video storage service ]

## Config files

* App static config files are copied into the container at build time
* Media server config file to use on deploy can be found at `configs/mediamtx/mediamtx.yaml`
* MQTT server config file to use on deploy can be found at `configs/mosquitto/mosquitto.conf`


## ENV variables
```
# caps how often alerts are generated
ALERTS_COOLDOWN_SECONDS=1.0
# list of (longitude, latitude) couples, or 'not_specified' to deactivate geofencing
GEOFENCING_VERTEXES=(24.173920, 35.427231),(24.174090, 35.427115),(24.174063, 35.427041),(24.173813, 35.427096)
# size of the safety radius around animals
SAFETY_RADIUS_M=2.0
# slope inclinations after which the area is marked as dangerous
SLOPE_ANGLE_THRESHOLD=30.0
# how many seconds of tracking to consider for anomaly detection
SLIDING_WINDOW_SIZE_S=30.0

# drone camera parameters (depends on drone model)
DRONE_TRUE_FOCAL_LEN_MM=12.29
DRONE_SENSOR_WIDTH_MM=17.35
DRONE_SENSOR_HEIGHT_MM=13.00
DRONE_SENSOR_WIDTH_PIXELS=5280
DRONE_SENSOR_HEIGHT_PIXELS=3956

# connecting to the mediaserver to retrieve the input video stream, stream key='drone'
# if the set of containers is isolated, then authentication is not required, and protocol/ip/port are fixed.
VIDEO_STREAM_READER_USERNAME=not_specified
VIDEO_STREAM_READER_PASSWORD=not_specified
VIDEO_STREAM_READER_HOST=172.17.0.1
VIDEO_STREAM_READER_PROTOCOL=rtsp
VIDEO_STREAM_READER_PORT=8554
VIDEO_STREAM_READER_STREAM_KEY=drone

# connecting to the mqtt server to retrieve the input telemetry
# if the set of containers is isolated, then authentication is not required, and protocol/ip/port are fixed.
TELEMETRY_LISTENER_USERNAME=not_specified
TELEMETRY_LISTENER_PASSWORD=not_specified
TELEMETRY_LISTENER_HOST=172.17.0.1
TELEMETRY_LISTENER_PROTOCOL=mqtt
TELEMETRY_LISTENER_PORT=1883

# Websocket server internal to the app. 
# UI connects to receives alerts. 
# decide which port to expose to the UI
WEBSOCKET_HOST=0.0.0.0 # fixed
WEBSOCKET_PORT=8443

# Connecting to the AGRARIAN DB
# Worker is the single account with read/write permission, avoid having to give each user permissions to the DB
# verifies that user exists and is valid, then updates DB on user's behalf
DB_WORKER_NAME=app_manager
DB_WORKER_PASSWORD=app_manager_pass
DB_USERNAME=testuser@testmail.com
DB_PASSWORD=testpassword
DB_HOST=172.17.0.1
DB_SERVICE=postgresql
DB_PORT=5432

# push annotated video stream to the mediaserver so that user/UI can see it
VIDEO_OUT_STREAM_USERNAME=not_specified
VIDEO_OUT_STREAM_PASSWORD=not_specified
VIDEO_OUT_STREAM_HOST=172.17.0.1
VIDEO_OUT_STREAM_PROTOCOL=rtmp
VIDEO_OUT_STREAM_PORT=1935
VIDEO_OUT_STREAM_STREAM_KEY=annot

# Video storage service
VIDEO_OUT_STORE_USERNAME=not_specified
VIDEO_OUT_STORE_PASSWORD=not_specified
VIDEO_OUT_STORE_HOST=0.0.0.0
VIDEO_OUT_STORE_SERVICE=azure
VIDEO_OUT_STORE_PORT=443
```


# DEMO

tab 1
```
source launch_mediamtx.sh && docker logs -f mediamtx_server
```

tab 2
```
source simulation/simulate_video_stream_rtmp_v2.sh
```

tab 3
```
source launch_mosquitto.sh && docker exec -it mosquitto mosquitto_sub -t "telemetry/#" -v
``` 

tab 4
```
source .venv/bin/activate && python3 simulation/telemetry.py
``` 

tab 5
```
source launch_postgres.sh
``` 

tab 6
```
source launch_danger_detection.sh -b -r
``` 

tab 7
```
source launch_ui.sh --webrtc-host 172.17.0.1 --ws-host 172.17.0.1
``` 

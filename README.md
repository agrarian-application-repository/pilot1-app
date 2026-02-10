![AGRARIAN](assets/agrarian.png)


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

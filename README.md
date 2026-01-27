![AGRARIAN](assets/agrarian.png)

# AGRARIAN: Herd Monitoring

## Setup
1.  Copy the project with
 ```
 git clone https://github.com/simonesarti/AGRARIAN.git
 ```
2. Enter the project with
 ```
 cd AGRARIAN
 ```
3. Create a file `.env` with the following content:
 ```
 DOCKER_USERNAME="$oauthtoken"
 DOCKER_PASSWORD="<your-ncvr-io-token>"
 ```


## Developing with virtual environments (conda)
4. Move to the `dev` folder with
 ```
 cd dev
``` 

5. Create the conda environment with
 ```
 bash create_conda_env.sh
 ``` 

6. Run experiments with:
 ```
bash run_script.sh <script.py> [--arg1 arg1val ... --argN argNval]
 ```  

# Docker

## Health monitoring
1. Enter the project with
 ```
 cd AGRARIAN
 ```

2. Build the Docker image with
 ```bash
docker build -t health_monitoring -f docker/health_monitoring/Dockerfile .
```

3. Verify the docker image was built succesfully
```
docker images | grep health_monitoring
```

4. Run the containerized application with
 ```
docker run --rm \
-e <http://ip:port/video>
-v </your/path/to/config_file.yaml>:/app/config.yaml \
health_monitoring config.yaml
```


create dokcer network agrarian-network

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list



curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit

sudo systemctl restart docker


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

tab 5
```
source .venv/bin/activate && python3 simulation/telemetry.py
``` 

tab 6
```
source launch_postgres.sh
``` 

tab 7
```
source launch_danger_detection.sh -b -r
``` 

tab8
```
source launch_ui.sh --webrtc-host 172.17.0.1 --ws-host 172.17.0.1
``` 

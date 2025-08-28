#!/bin/bash

# Ensure no existing container with the same name
docker rm -f mediamtx_server || true

echo "Starting MediaMTX Docker container..."


# port 8554/tcp and 8554/udp for RTSP streams are handles internally by the docker network, no need for external mapping
docker run -d \
  --name mediamtx_server \
  --network agrarian-network \
  -p "1935:1935/tcp" \
  -p "8889:8889/tcp" \
  -v "$(pwd)/configs/mediamtx/my_mediamtx.yaml:/mediamtx.yml:ro" \
  bluenviron/mediamtx:latest /mediamtx.yml

if [ $? -eq 0 ]; then
    echo "MediaMTX container 'mediamtx_server' started successfully."
    echo "RTMP is available on port 1935"
    echo "RTSP is available on port 8554"
    echo "WEBRTC is available on port 8889"
    echo "You can check its logs with: docker logs -f mediamtx_server"
else
    echo "Failed to start MediaMTX container."
fi

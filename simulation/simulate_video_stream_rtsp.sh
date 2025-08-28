#!/bin/bash

# Set up the RTMP server details (where FFmpeg will push the stream to)
# Note: The stream key 'stream' is appended to the URL
rtsp_stream_url="rtsp://127.0.0.1:8554/annot"

echo "Starting RTSP stream to: $rtsp_stream_url"
echo "Make sure your RTMP server is running on port 8554"
echo "Press Ctrl+C to stop streaming"

# Continuously stream the video in loop using FFmpeg
# Restart FFmpeg if it crashes or the connection drops
while true; do
    echo "Starting FFmpeg stream push..."
    ffmpeg \
    -re \
    -stream_loop -1 \
    -i rainbow_transition.mp4 \
    -c:v libx264 \
    -preset ultrafast \
    -tune zerolatency \
    -s 1920x1080 \
    -pix_fmt yuv420p \
    -r 30 \
    -g 60 \
    -keyint_min 60 \
    -sc_threshold 0 \
    -b:v 2500k \
    -maxrate 2500k \
    -bufsize 5000k \
    -f rtsp \
    -rtsp_transport tcp \
    ${rtsp_stream_url}
    
    exit_code=$?
    echo "FFmpeg exited with code: $exit_code"
    
    if [ $exit_code -eq 0 ]; then
        echo "Stream ended normally"
        break
    else
        echo "FFmpeg stream push failed. Retrying in 5 seconds..."
        sleep 5
    fi
done

echo "Streaming stopped"


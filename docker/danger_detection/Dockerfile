# Use an NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:25.01-py3

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt update && apt install -y \
    ffmpeg \
    libx264-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \ 
    libswscale-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies without caching
RUN pip install --no-cache-dir -r requirements.txt

# Make additional directories:

# logs (for debugging)
RUN mkdir -p /app/logs && chmod 777 /app/logs

# outputs
RUN mkdir -p /app/outputs && chmod 777 /app/outputs

# DEM directory
RUN mkdir -p /app/dem && chmod 777 /app/dem
# model checkpoints
RUN mkdir -p /app/checkpoints && chmod +r /app/checkpoints
COPY checkpoints/danger_detection/ /app/checkpoints/danger_detection
# configs
RUN mkdir -p /app/configs/danger_detection && chmod +r /app/configs/danger_detection

# src
RUN mkdir -p /app/src && chmod 777 /app/src
RUN mkdir -p /app/src/configs/danger_detection && chmod 777 /app/src/configs/danger_detection
COPY src/configs/danger_detection.py /app/src/configs/
COPY src/danger_detection /app/src/danger_detection
COPY src/shared /app/src/shared/
# main
COPY danger_detection.py .

# Input ports:
# 1935/tcp: For RTMP input
# 12345/udp: Your custom UDP input
EXPOSE 1935/tcp 12345/udp 

# Output ports:
# 8554/tcp: RTSP control port
# 8554/udp: RTSP RTP data port (often used in conjunction with TCP for RTSP)
# 54321/tcp: custom TCP output
EXPOSE 8554/tcp 8554/udp 54321/tcp

# Set entrypoint
ENTRYPOINT ["python", "danger_detection_stream.py"]
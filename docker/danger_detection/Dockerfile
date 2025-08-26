# Use an NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:25.01-py3

ENV DEBIAN_FRONTEND=noninteractive

ENV STREAM_IP="127.0.0.1"
ENV STREAM_PORT="1935"
ENV STREAM_NAME="drone"
ENV TELEMETRY_IP="0.0.0.0"
ENV TELEMETRY_PORT="12345"
ENV ANNOTATIONS_IP="127.0.0.1"
ENV ANNOTATIONS_PORT="8554"
ENV ANNOTATIONS_NAME="annot"
ENV ALERTS_IP="127.0.0.1"
ENV ALERTS_PORT="54321"

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
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker caching
COPY docker/danger_detection/requirements.txt ./requirements.txt

# Install Python dependencies without caching
RUN pip install --no-cache-dir -r requirements.txt

# Create first level directories
RUN mkdir -p /app/{logs,outputs,dem,checkpoints,configs,src}
# Create inner directories
RUN mkdir -p /app/{configs/danger_detection,src/configs/danger_detection}

# Copy checkpoints from repo to container 
COPY checkpoints/danger_detection/ /app/checkpoints/danger_detection
# Copy useful arguments checking code to container 
COPY src/configs/danger_detection.py src/configs/drone.py src/configs/networks.py /app/src/configs/
# Copy danger detection only code to container
COPY src/danger_detection /app/src/danger_detection
# Copy danger detection shared (with health monitoring) code to container
COPY src/shared /app/src/shared/
# Copy main script to container
COPY danger_detection_stream.py .

# Set permissions for entire /app directory at once
RUN chmod -R 777 /app

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
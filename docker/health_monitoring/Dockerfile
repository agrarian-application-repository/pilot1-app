FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

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
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker caching
COPY docker/health_monitoring/requirements.txt ./requirements.txt

# Install Python dependencies without caching
RUN pip install --no-cache-dir -r requirements.txt

# Create first level directories
RUN mkdir -p /app/{logs,outputs,dem,checkpoints,configs,src}
# Create inner directories
RUN mkdir -p /app/{configs/health_monitoring,src/configs/health_monitoring}

# Copy checkpoints from repo to container 
COPY checkpoints/health_monitoring/ /app/checkpoints/health_monitoring
# Copy useful arguments checking code to container 
COPY src/configs/health_monitoring.py src/configs/drone.py src/configs/networks.py src/configs/utils.py /app/src/configs/
# Copy health monitoring only code to container
COPY src/health_monitoring /app/src/health_monitoring
# Copy health monitoring shared (with danger detection) code to container
COPY src/shared /app/src/shared/
# Copy main script to container
COPY health_monitoring_stream.py .

# Set permissions for entire /app directory at once
RUN chmod -R 777 /app

ENV STREAM_IP="mediamtx_server"
ENV STREAM_PORT="1935"
ENV STREAM_NAME="drone"
ENV TELEMETRY_IP="0.0.0.0"
ENV TELEMETRY_PORT="12345"
ENV ANNOTATIONS_IP="mediamtx_server"
ENV ANNOTATIONS_PORT="8554"
ENV ANNOTATIONS_NAME="annot"
ENV ALERTS_IP="10.91.222.62"
ENV ALERTS_PORT="54321"

# Input ports:
# 1935/tcp: For RTMP input (not exposed, handled by the mediamtx server)
# 12345/udp: Your custom UDP input
EXPOSE 12345/udp 

# Output ports:
# 8554/tcp: RTSP control port (not exposed, handled internally by the docker network)
# 8554/udp: RTSP RTP data port (not exposed, handled internally by the docker network)
# 54321/tcp: custom TCP output (not exposed, application is the sender not the receiver)

# Set entrypoint
ENTRYPOINT ["python", "health_monitoring_stream.py"]
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

# Make additional directories
RUN mkdir -p /app/outputs && chmod 777 /app/outputs
RUN mkdir -p /app/logs && chmod 777 /app/logs


# Copy application code into the container:

# model checkpoints
RUN mkdir -p /app/checkpoints && chmod +r /app/checkpoints
COPY checkpoints/health_monitoring/ /app/checkpoints/health_monitoring
# configs
RUN mkdir -p /app/configs/health_monitoring && chmod +r /app/configs/health_monitoring
COPY configs/health_monitoring/ /app/configs/health_monitoring
COPY configs/drone_specs.yaml /app/configs
# src
RUN mkdir -p /app/src && chmod 777 /app/src
RUN mkdir -p /app/src/configs/health_monitoring
COPY src/configs/health_monitoring.py /app/src/configs/
COPY src/health_monitoring /app/src/health_monitoring
COPY src/shared /app/src/shared/
# main
COPY health_monitoring.py .

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
ENTRYPOINT ["python", "health_monitoring.py"]
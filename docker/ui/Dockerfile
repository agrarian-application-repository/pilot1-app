# Use a simple Python runtime as image
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

# set default environmental variables
ENV TCP_PORT=54321
ENV MEDIAMTX_WEBRTC_URL="http://mediamtx:8889"
ENV STREAM_NAME="annot"
ENV STUN_SERVER="stun:stun.l.google.com:19302"

# Install system dependencies
RUN apt update && apt install -y \
    ffmpeg \
    libx264-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY docker/ui/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make additional directories
RUN mkdir -p /app/{logs,assets,src,src/ui}

# Copy the application script inside /app
COPY assets /app/assets
COPY src/ui /app/src/ui

# copy main file
COPY ui.py .

# Set permissions for entire /app directory at once
RUN chmod -R 777 /app

# Input port:
# 54321/tcp: custom TCP output for receiving alerts
# 8501/tcp: streamlit web UI
EXPOSE 54321/tcp
EXPOSE 8501/tcp

# Run the application
CMD ["streamlit", "run", "ui.py"]

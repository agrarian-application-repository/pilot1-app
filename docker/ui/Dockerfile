# Use a simple Python runtime as image
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

# set default environmental variables
ENV TCP_PORT=54321
ENV MEDIAMTX_WEBRTC_URL="http://mediamtx:8889"
ENV STREAM_NAME="annot"
ENV STUN_SERVER="stun:stun.l.google.com:19302"
ENV CONTAINER="true"

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
COPY ./docker/ui/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Make additional directories
RUN mkdir -p /app/logs && chmod 777 /app/logs
RUN mkdir -p /app/assets && chmod 777 /app/assets

# Copy the application script inside /app
COPY ./src/ui/* /app/
COPY ./assets/* /app/assets/

# Input port:
# 54321/tcp: custom TCP output
# 8501/tcp: streamlit web UI
EXPOSE 54321/tcp
EXPOSE 8501/tcp

# Run the application
CMD ["streamlit", "run", "ui.py"]

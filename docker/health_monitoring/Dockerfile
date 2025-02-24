# Use an NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:25.01-py3

# Import packages for video processing and streaming
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies without caching
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories in the container
RUN mkdir -p src/configs
RUN mkdir -p src/health_monitoring

# Copy application code into the container: configs
COPY src/configs/utils.py ./src/configs
COPY src/configs/health_monitoring.py ./src/configs
# Copy application code into the container: application
COPY src/health_monitoring ./src/health_monitoring
# Copy application code into the container: run script
COPY health_monitoring.py .

# Copy entrypoint and ensure entrypoint script is executable
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Set entrypoint
ENTRYPOINT ["./entrypoint.sh"]
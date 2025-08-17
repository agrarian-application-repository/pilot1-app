#!/bin/bash

# Danger detection Container Launch Script

# set -e  # Exit on any error

# Default values
IMAGE_NAME="agrarian-danger-detection"
CONTAINER_NAME="agrarian-danger-detection"

DEM_PATH=""
DEM_MASK_PATH=""

VIDEO_STREAM_ADDR=""
TELEMETRY_STREAM_ADDR=""

VIDEO_STREAM_OUT_ADDR=""
ALERTS_STREAM_OUT_ADDR=""

DETACHED="false"
REMOVE_EXISTING="false"
ENV_FILE=""
NETWORK=""
BUILD="false"

# Help function
show_help() {
    cat << EOF
Streamlit UI Container Launch Script

Usage: $0 [OPTIONS]

OPTIONS:
    -i, --image NAME                    Docker image name (default: agrarian-danger-detection)
    -n, --name NAME                     Container name (default: agrarian-danger-detection)
    -s, --stream_in URL                 Media server URL receiving the drone video stream (default: )
    -t, --telemetry URL                 Telemetry source URL (default: )
    -a, --stream_out URL
    --internal-tcp-port PORT            Internal TCP port (default: 54321)
    -m, --mediamtx-url URL              MediaMTX WebRTC URL (default: http://mediamtx:8889)
    --stream-name NAME                  Stream name (default: annot)
    --stun-server SERVER                STUN server (default: stun:stun.l.google.com:19302)
    -d, --detached                      Run in detached mode
    -r, --remove                        Remove existing container if it exists
    -f, --env-file FILE                 Load environment variables from file
    --network NETWORK                   Connect to specific Docker network
    -b, --build                         Build image before running
    -h, --help                          Show this help message

EXAMPLES:
    # Basic launch
    $0

    # Custom ports and stream
    $0 -p 55555 -s 8502 --stream-name drone

    # Production mode with env file
    $0 -d -r -f production.env --network my-network

    # Development with rebuild
    $0 -b -r --stream-name dev-stream

    # Custom MediaMTX connection
    $0 -m http://192.168.1.100:8889 --stream-name drone
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -n|--name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        -p|--tcp-port)
            HOST_TCP_PORT="$2"
            shift 2
            ;;
        -s|--streamlit-port)
            HOST_STREAMLIT_PORT="$2"
            shift 2
            ;;
        --internal-tcp-port)
            TCP_PORT="$2"
            shift 2
            ;;
        -m|--mediamtx-url)
            MEDIAMTX_WEBRTC_URL="$2"
            shift 2
            ;;
        --stream-name)
            STREAM_NAME="$2"
            shift 2
            ;;
        --stun-server)
            STUN_SERVER="$2"
            shift 2
            ;;
        -d|--detached)
            DETACHED="true"
            shift
            ;;
        -r|--remove)
            REMOVE_EXISTING="true"
            shift
            ;;
        -f|--env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        --network)
            NETWORK="$2"
            shift 2
            ;;
        -b|--build)
            BUILD="true"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Build image if requested
if [[ "$BUILD" == "true" ]]; then
    echo "Building Docker image: $IMAGE_NAME"
    docker build -f "./docker/ui/Dockerfile" -t "$IMAGE_NAME" .
fi

# Remove existing container if requested
if [[ "$REMOVE_EXISTING" == "true" ]]; then
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Removing existing container: $CONTAINER_NAME"
        docker rm -f "$CONTAINER_NAME"
    fi
fi

# Check if container already exists and is running
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container $CONTAINER_NAME is already running!"
    echo "Use -r/--remove flag to remove and recreate, or choose a different name."
    exit 1
fi

# Build docker run command
DOCKER_CMD="docker run"

# Add detached mode if requested
if [[ "$DETACHED" == "true" ]]; then
    DOCKER_CMD="$DOCKER_CMD -d"
else
    DOCKER_CMD="$DOCKER_CMD -it"
fi

# Add container name
DOCKER_CMD="$DOCKER_CMD --name $CONTAINER_NAME"

# Add port mappings
DOCKER_CMD="$DOCKER_CMD -p $HOST_TCP_PORT:$TCP_PORT"
DOCKER_CMD="$DOCKER_CMD -p $HOST_STREAMLIT_PORT:8051"

# Add network if specified
if [[ -n "$NETWORK" ]]; then
    DOCKER_CMD="$DOCKER_CMD --network $NETWORK"
fi

# Add environment variables
DOCKER_CMD="$DOCKER_CMD -e TCP_PORT=$TCP_PORT"
DOCKER_CMD="$DOCKER_CMD -e MEDIAMTX_WEBRTC_URL=$MEDIAMTX_WEBRTC_URL"
DOCKER_CMD="$DOCKER_CMD -e STREAM_NAME=$STREAM_NAME"
DOCKER_CMD="$DOCKER_CMD -e STUN_SERVER=$STUN_SERVER"

# Map logs folder
DOCKER_CMD="$DOCKER_CMD -v $(pwd)/logs:/app/logs"


# Add env file if specified
if [[ -n "$ENV_FILE" ]]; then
    if [[ -f "$ENV_FILE" ]]; then
        DOCKER_CMD="$DOCKER_CMD --env-file $ENV_FILE"
    else
        echo "Environment file not found: $ENV_FILE"
        exit 1
    fi
fi

# Add image name
DOCKER_CMD="$DOCKER_CMD $IMAGE_NAME"

# Display configuration
echo "================================================"
echo "Streamlit UI Container Configuration"
echo "================================================"
echo "Image Name:           $IMAGE_NAME"
echo "Container Name:       $CONTAINER_NAME"
echo "Host TCP Port:        $HOST_TCP_PORT"
echo "Host Streamlit Port:  $HOST_STREAMLIT_PORT"
echo "Internal TCP Port:    $TCP_PORT"
echo "MediaMTX URL:         $MEDIAMTX_WEBRTC_URL"
echo "Stream Name:          $STREAM_NAME"
echo "STUN Server:          $STUN_SERVER"
echo "Detached Mode:        $DETACHED"
echo "Network:              ${NETWORK:-default}"
echo "Environment File:     ${ENV_FILE:-none}"
echo "================================================"

# Ask for confirmation unless in detached mode
if [[ "$DETACHED" != "true" ]]; then
    read -p "Launch container with these settings? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Launch cancelled."
        exit 0
    fi
fi

# Execute docker run command
echo "Launching container..."
echo "Command: $DOCKER_CMD"
echo ""

exec $DOCKER_CMD
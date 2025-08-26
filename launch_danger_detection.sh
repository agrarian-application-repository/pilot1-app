#!/bin/bash

# Danger detection Container Launch Script

# set -e  # Exit on any error

# Default values
IMAGE_NAME="agrarian-danger-detection-stream"
CONTAINER_NAME="agrarian-danger-detection-stream"

DEM_PATH=""
DEM_MASK_PATH=""

# Algorithms configuration
INPUT_CONFIG=""
OUTPUT_CONFIG=""
DRONE_CONFIG=""
DETECTOR_CONFIG=""
SEGMENTER_CONFIG=""

# Stream and network configuration variables
STREAM_IP=""
STREAM_PORT=""
STREAM_NAME=""
TELEMETRY_IP=""
TELEMETRY_PORT=""
ANNOTATIONS_IP=""
ANNOTATIONS_PORT=""
ANNOTATIONS_NAME=""
ALERTS_IP=""
ALERTS_PORT=""

DETACHED="false"
REMOVE_EXISTING="false"
ENV_FILE=""
NETWORK=""
BUILD="false"

# Help function
show_help() {
    cat << EOF
Agrarian Danger Detection Stream Container Launch Script

Usage: $0 [OPTIONS]

OPTIONS:
    -i, --image         NAME                    Docker image name (default: agrarian-danger-detection-stream)
    -n, --name          NAME                    Container name (default: agrarian-danger-detection-stream)
    --dem               TIF                     DEM data
    --dem_mask          TIF                     DEM mask data
    --in_conf           YAML                    Input config file
    --drone_conf        YAML                    Drone config file
    
    --stream_ip         URL(IP)                 IP of the mediamtx server from which to retrieve the stream (default: 127.0.0.1)
    --stream_port       URL(PORT)               Port where to receive the stream from the mediamtx server (exposed Dockerfile: 1935, RTMP)
    --stream_name       URL(NAME)               Name of the stream (default: drone)
    --telemetry_ip      URL(IP)                 IP where to receive the telemetry packets (default: 0.0.0.0 - listen on all interfaces)
    --telemetry_port    URL(PORT)               Port where to receive the UDP packets (exposed Dockerfile: 12345/udp)
    --annotations_ip    URL(IP)                 IP of the mediamtx server where to send the annotated stream (default: 127.0.0.1)
    --annotations_port  URL(PORT)               Port where to send the annotated stream (exposed Dockerfile: 85547/tcp 8554/udp, RTSP)
    --annotations_name  URL(NAME)               Name of the annotated stream (default: annot)
    --alerts_ip         URL(IP)                 IP of the machine where to send the alerts TPC packets (default: 127.0.0.1)
    --alerts_port       URL(PORT)               Port where to send the alerts TCP packets (exposed Dockerfile: 54321/tcp)

    -d, --detached                              Run in detached mode
    -r, --remove                                Remove existing container if it exists
    -f, --env-file      FILE                    Load environment variables from file
    --network           NETWORK                 Connect to specific Docker network
    -b, --build                                 Build image before running
    -h, --help                                  Show this help message

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
        --dem)
            DEM_PATH="$2"
            if [[ -f "$DEM_PATH" ]] && [[ "$DEM_PATH" == *.tif ]]; then
                shift 2
            else
                echo "Error: --dem must be an existing .tif file."
                exit 1
            fi
            ;;
        --dem_mask)
            DEM_MASK_PATH="$2"
            if [[ -f "$DEM_MASK_PATH" ]] && [[ "$DEM_MASK_PATH" == *.tif ]]; then
                shift 2
            else
                echo "Error: --dem_mask must be an existing .tif file."
                exit 1
            fi
            ;;
        --in_conf)
            INPUT_CONFIG="$2"
            if [[ -f "$INPUT_CONFIG" ]] && [[ "$INPUT_CONFIG" == *.yaml ]]; then
                shift 2
            else
                echo "Error: --in_conf requires an existing .yaml file."
                exit 1
            fi
            ;;
        --drone_conf)
            DRONE_CONFIG="$2"
            if [[ -f "$DRONE_CONFIG" ]] && [[ "$DRONE_CONFIG" == *.yaml ]]; then
                shift 2
            else
                echo "Error: --drone_conf requires an existing .yaml file."
                exit 1
            fi
            ;;
        --stream_ip)
            STREAM_IP="$2"
            shift 2
            ;;
        --stream_port)
            STREAM_PORT="$2"
            shift 2
            ;;
        --stream_name)
            STREAM_NAME="$2"
            shift 2
            ;;
        --telemetry_ip)
            TELEMETRY_IP="$2"
            shift 2
            ;;
        --telemetry_port)
            TELEMETRY_PORT="$2"
            shift 2
            ;;
        --annotations_ip)
            ANNOTATIONS_IP="$2"
            shift 2
            ;;
        --annotations_port)
            ANNOTATIONS_PORT="$2"
            shift 2
            ;;
        --annotations_name)
            ANNOTATIONS_NAME="$2"
            shift 2
            ;;
        --alerts_ip)
            ALERTS_IP="$2"
            shift 2
            ;;
        --alerts_port)
            ALERTS_PORT="$2"
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

# Check for mandatory arguments after parsing is complete.
# This ensures all required options were provided.
if [[ -z "$INPUT_CONFIG" || -z "$DRONE_CONFIG" ]]; then
    echo "Error: Missing one or more required configuration files."
    show_help
    exit 1
fi

# Build image if requested
if [[ "$BUILD" == "true" ]]; then
    echo "Building Docker image: $IMAGE_NAME"
    docker build -f "./docker/danger_detection/Dockerfile" -t "$IMAGE_NAME" .
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

# Stream input (RTMP) - default port 1935
if [[ -n "$STREAM_PORT" ]]; then
    DOCKER_CMD="$DOCKER_CMD -p $STREAM_PORT:1935"
else
    DOCKER_CMD="$DOCKER_CMD -p 1935:1935"
fi

# Telemetry input (UDP) - default port 12345
if [[ -n "$TELEMETRY_PORT" ]]; then
    DOCKER_CMD="$DOCKER_CMD -p $TELEMETRY_PORT:12345/udp"
else
    DOCKER_CMD="$DOCKER_CMD -p 12345:12345/udp"
fi

# Annotations output (RTSP) - default ports 8554 (UDP) and 8554 (TCP)
if [[ -n "$ANNOTATIONS_PORT" ]]; then
    DOCKER_CMD="$DOCKER_CMD -p $ANNOTATIONS_PORT:8554/tcp -p $ANNOTATIONS_PORT:8554/udp"
else
    DOCKER_CMD="$DOCKER_CMD -p 8554:8554/tcp -p 8554:8554/udp"
fi

# Alerts output (TCP) - default port 54321
if [[ -n "$ALERTS_PORT" ]]; then
    DOCKER_CMD="$DOCKER_CMD -p $ALERTS_PORT:54321/tcp"
else
    DOCKER_CMD="$DOCKER_CMD -p 54321:54321/tcp"
fi


# Add network if specified
if [[ -n "$NETWORK" ]]; then
    DOCKER_CMD="$DOCKER_CMD --network $NETWORK"
fi


# Add environment variables (IPs/Ports/Names)
if [[ -n "$STREAM_IP" ]]; then
    DOCKER_CMD="$DOCKER_CMD -e STREAM_IP=$STREAM_IP"
fi

if [[ -n "$STREAM_PORT" ]]; then
    DOCKER_CMD="$DOCKER_CMD -e STREAM_PORT=$STREAM_PORT"
else
    DOCKER_CMD="$DOCKER_CMD -e STREAM_PORT=1935"
fi

if [[ -n "$STREAM_NAME" ]]; then
    DOCKER_CMD="$DOCKER_CMD -e STREAM_NAME=$STREAM_NAME"
fi

# For telemetry receiving, default to 0.0.0.0 to listen on all interfaces
if [[ -n "$TELEMETRY_IP" ]]; then
    DOCKER_CMD="$DOCKER_CMD -e TELEMETRY_IP=$TELEMETRY_IP"
else
    DOCKER_CMD="$DOCKER_CMD -e TELEMETRY_IP=0.0.0.0"
fi

if [[ -n "$TELEMETRY_PORT" ]]; then
    DOCKER_CMD="$DOCKER_CMD -e TELEMETRY_PORT=$TELEMETRY_PORT"
else
    DOCKER_CMD="$DOCKER_CMD -e TELEMETRY_PORT=12345"
fi

if [[ -n "$ANNOTATIONS_IP" ]]; then
    DOCKER_CMD="$DOCKER_CMD -e ANNOTATIONS_IP=$ANNOTATIONS_IP"
fi

if [[ -n "$ANNOTATIONS_PORT" ]]; then
    DOCKER_CMD="$DOCKER_CMD -e ANNOTATIONS_PORT=$ANNOTATIONS_PORT"
else
    DOCKER_CMD="$DOCKER_CMD -e ANNOTATIONS_PORT=8554"
fi

if [[ -n "$ANNOTATIONS_NAME" ]]; then
    DOCKER_CMD="$DOCKER_CMD -e ANNOTATIONS_NAME=$ANNOTATIONS_NAME"
fi

if [[ -n "$ALERTS_IP" ]]; then
    DOCKER_CMD="$DOCKER_CMD -e ALERTS_IP=$ALERTS_IP"
fi

if [[ -n "$ALERTS_PORT" ]]; then
    DOCKER_CMD="$DOCKER_CMD -e ALERTS_PORT=$ALERTS_PORT"
else
    DOCKER_CMD="$DOCKER_CMD -e ALERTS_PORT=54321"
fi


# Map to local logs folder for debugging
DOCKER_CMD="$DOCKER_CMD -v $(pwd)/logs:/app/logs"

# Map outputs folder to local to get access to results
DOCKER_CMD="$DOCKER_CMD -v $(pwd)/outputs:/app/outputs"

# Add volume mapping for DEM if it's set
if [[ -n "$DEM_PATH" ]]; then
    DOCKER_CMD="$DOCKER_CMD -v $DEM_PATH:/app/dem/dem.tif"
fi

# Add volume mapping for DEM_MASK if it's set
if [[ -n "$DEM_MASK_PATH" ]]; then
    DOCKER_CMD="$DOCKER_CMD -v $DEM_MASK_PATH:/app/dem/dem_mask.tif"
fi

# Add config files via volume mapping
DOCKER_CMD="$DOCKER_CMD -v $(pwd)/$INPUT_CONFIG:/app/configs/danger_detection/input.yaml"
DOCKER_CMD="$DOCKER_CMD -v $(pwd)/$DRONE_CONFIG:/app/configs/drone_specs.yaml"
DOCKER_CMD="$DOCKER_CMD -v $(pwd)/configs/danger_detection/detector.yaml:/app/configs/danger_detection/detector.yaml"
DOCKER_CMD="$DOCKER_CMD -v $(pwd)/configs/danger_detection/segmenter.yaml:/app/configs/danger_detection/segmenter.yaml"
DOCKER_CMD="$DOCKER_CMD -v $(pwd)/configs/danger_detection/output.yaml:/app/configs/danger_detection/output.yaml"


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
echo "======================================================="
echo "Agrarian Danger Detection StreamContainer Configuration"
echo "======================================================="
echo "Image Name:           $IMAGE_NAME"
echo "Container Name:       $CONTAINER_NAME"
echo "Detached Mode:        $DETACHED"
echo "Network:              ${NETWORK:-default}"
echo "Environment File:     ${ENV_FILE:-none}"
echo ""
echo "Stream Configuration:"
echo "  Input Stream IP:    ${STREAM_IP:-127.0.0.1 (default)}"
echo "  Input Stream Port:  ${STREAM_PORT:-1935 (default)}"
echo "  Stream Name:        ${STREAM_NAME:-drone (default)}"
echo ""
echo "Telemetry Configuration:"
echo "  Telemetry IP:       ${TELEMETRY_IP:-0.0.0.0 (default - listen on all interfaces)}"
echo "  Telemetry Port:     ${TELEMETRY_PORT:-12345 (default)}"
echo ""
echo "Annotations Configuration:"
echo "  Annotations IP:     ${ANNOTATIONS_IP:-127.0.0.1 (default)}"
echo "  Annotations Port:   ${ANNOTATIONS_PORT:-8554 (default)}"
echo "  Annotations Name:   ${ANNOTATIONS_NAME:-annot (default)}"
echo ""
echo "Alerts Configuration:"
echo "  Alerts IP:          ${ALERTS_IP:-127.0.0.1 (default)}"
echo "  Alerts Port:        ${ALERTS_PORT:-54321 (default)}"
echo "======================================================"

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

# exec $DOCKER_CMD
$DOCKER_CMD

# source launch_danger_detection.sh -b -d -r --in_conf configs/danger_detection/input.yaml --drone_conf configs/drone_specs.yaml
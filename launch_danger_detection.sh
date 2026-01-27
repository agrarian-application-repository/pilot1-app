#!/bin/bash

# Dnager Detection Container Launch Script

# set -e  # Exit on any error

# paths for docker build
DOCKERFILE_PATH="./docker/danger_detection/Dockerfile"
DOCKERIGNORE_PATH="./docker/danger_detection/.dockerignore"
REQUIREMENTS_PATH="./docker/danger_detection/requirements.txt"
ROOT_DOCKERIGNORE_PATH="./.dockerignore"
ROOT_REQUIREMENTS_PATH="./requirements.txt"

# Default values
IMAGE_NAME="agrarian-dd"
CONTAINER_NAME="agrarian-dd"
DETACHED="false"
REMOVE_EXISTING="true"
ENV_FILE="$(pwd)/docker/danger_detection/.env"
NETWORK=""
BUILD="false"

DEM_PATH=""
DEM_MASK_PATH=""

# Help function
show_help() {
    cat << EOF
Agrarian Danger Detection Container Launch Script

Usage: $0 [OPTIONS]

OPTIONS:
    -i, --image         NAME                    Docker image name (default: agrarian-danger-detection-stream)
    -n, --name          NAME                    Container name (default: agrarian-danger-detection-stream)
    -d, --detached                              Run in detached mode
    -r, --remove                                Remove existing container if it exists
    -f, --env-file      FILE                    Load environment variables from file
    --network           NETWORK                 Connect to specific Docker network
    -b, --build                                 Build image before running
    -h, --help                                  Show this help message

    --dem               TIF                     DEM data
    --dem_mask          TIF                     DEM mask data
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
    # copy .dockerignore to context root for build
    cp "$DOCKERIGNORE_PATH" "$ROOT_DOCKERIGNORE_PATH"
    # copy requirements.txt to context root for build
    cp "$REQUIREMENTS_PATH" "$ROOT_REQUIREMENTS_PATH"
    # build docker image
    docker build -f "$DOCKERFILE_PATH" -t "$IMAGE_NAME" .
    # remove .dockerignore copy from context root
    rm "$ROOT_DOCKERIGNORE_PATH"
    # remove requirements.txt copy from context root
    rm "$ROOT_REQUIREMENTS_PATH"
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

# use gpus
DOCKER_CMD="$DOCKER_CMD --gpus all"

# Add container name
DOCKER_CMD="$DOCKER_CMD --name $CONTAINER_NAME"

# Add port mappings
# port 8443 to allow Websocket requiest from UI
DOCKER_CMD="$DOCKER_CMD -p 8443:8443"

# Add network if specified
if [[ -n "$NETWORK" ]]; then
    DOCKER_CMD="$DOCKER_CMD --network $NETWORK"
fi

# Add environment variables
# ...
# skip, use file

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
DOCKER_CMD="$DOCKER_CMD -v $(pwd)/configs/danger_detection/detector.yaml:/app/configs/danger_detection/detector.yaml"
DOCKER_CMD="$DOCKER_CMD -v $(pwd)/configs/danger_detection/segmenter.yaml:/app/configs/danger_detection/segmenter.yaml"

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

# Display configuration (simplified - showing actual values)
echo "======================================================="
echo "Agrarian Danger Detection StreamContainer Configuration"
echo "======================================================="
echo "Image Name:           $IMAGE_NAME"
echo "Container Name:       $CONTAINER_NAME"
echo "Detached Mode:        $DETACHED"
echo "Network:              ${NETWORK:-default}"
echo "Environment File:     ${ENV_FILE:-none}"
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

# source launch_danger_detection.sh -r
# source launch_danger_detection.sh -b -r
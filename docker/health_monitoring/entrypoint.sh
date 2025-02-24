#!/bin/bash

# Fail on error
set -e

# Check if a config file argument was provided
if [ -z "$1" ]; then
    echo "Error: No config file provided."
    exit 1
fi

CONFIG_FILE=$1

# Ensure the config file exists in the container
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found in the container!"
    exit 1
fi

# Ensure VIDEO_SOURCE is set
if [ -z "$VIDEO_SOURCE" ]; then
    echo "ERROR: VIDEO_SOURCE is not set."
    exit 1
fi

# Run the application with the config file
exec python health_monitoring.py --config "$CONFIG_FILE"

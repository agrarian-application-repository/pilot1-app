#!/bin/bash

# Check if PROJECT_PATH is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <absolute/path/to/project>"
  exit 1
fi

# Set PROJECT_PATH from the first argument
export PROJECT_PATH=$1

# Source other environment variables from .env
source .env
export SINGULARITY_DOCKER_USERNAME
export SINGULARITY_DOCKER_PASSWORD

# Define Singularity build command
singularity_build_command="singularity build agrarian.sif singularity.def"

# Execute command
echo "BUILDING SINGULARITY IMAGE ..."
echo "$singularity_build_command"
eval "$singularity_build_command"

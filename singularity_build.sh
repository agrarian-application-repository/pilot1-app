#!/bin/bash

source .env
export PROJECT_PATH
export SINGULARITY_DOCKER_USERNAME
export SINGULARITY_DOCKER_PASSWORD

# Define Singularity build command
singularity_build_command="singularity build agrarian.sif singularity.def"

# Execute command
echo "BUILDING SINGULARITY IMAGE ..."
echo "$singularity_build_command"
eval "$singularity_build_command"

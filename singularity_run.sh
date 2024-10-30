#!/bin/bash

module load proxy/proxy_20

# Define Singularity run command
# everything in $@ is given to %runscript in singularity.def
singularity_run_command="
    singularity run \
    --nv \
    --bind ./data:/opt/app/data \
    --bind ./experiments:/opt/app/experiments \
    --bind ./src:/opt/app/src \
    --bind ./test:/opt/app/test \
    --bind ./configs:/opt/app/configs \
    agrarian.sif \
    $@
    "

# Run Singularity container
echo "RUNNING SINGULARITY CONTAINER ..."
echo "$singularity_run_command"
eval "$singularity_run_command"


# bash singularity_run.sh <script.py> [--arg1 arg1 --arg2 arg2]
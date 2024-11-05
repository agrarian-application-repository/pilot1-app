#!/bin/bash

##########################################################################

#PBS -S /bin/bash
#PBS -N "AGRARIAN_RUN "
#PBS -q gpu
#PBS -l select=1:ncpus=1:ngpus=1,walltime=04:00:00
#PBS -k eo
#PBS -j eo

##########################################################################

cd $PBS_O_WORKDIR

args="$ARGS"

bash singularity_run.sh $args

# qsub -v 'ARGS="<script.py> [--arg1 arg1 --arg2 arg2]"' pbs_singularity_run.sh

#!/bin/bash

##########################################################################

#PBS -S /bin/bash
#PBS -N "AGRARIAN_RUN"
#PBS -q gpu
#PBS -l select=1:ncpus=1:ngpus=1,walltime=04:00:00
#PBS -k eo
#PBS -j eo

##########################################################################

cd $PBS_O_WORKDIR

args="$ARGS"

bash run_script.sh $args

# qsub -v 'ARGS="<script.py> [--arg1 arg1 --arg2 arg2]"' pbs_run_script.sh


# qsub -q gpu -l select=1:ncpus=32:ngpus=4,walltime=08:00:00 -v 'ARGS="train.py --config configs/train_config_m_320.yaml"' pbs_run_script.sh
# qsub -q gpu -l select=1:ncpus=32:ngpus=4,walltime=08:00:00 -v 'ARGS="train.py --config configs/train_config_x_320.yaml"' pbs_run_script.sh
# qsub -q gpu -l select=1:ncpus=32:ngpus=4,walltime=08:00:00 -v 'ARGS="train.py --config configs/train_config_m_480crops_1280_720.yaml"' pbs_run_script.sh

# qsub -q gpu -l select=1:ncpus=48:ngpus=1,walltime=96:00:00 -v 'ARGS="hyperparameters_search.py --config configs/hs_search_1280_720.yaml"' pbs_run_script.sh

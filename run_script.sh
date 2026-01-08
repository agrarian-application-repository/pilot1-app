#!/bin/bash

module load proxy/proxy_16
module load cuda12.9/toolkit/12.9
module load miniconda3/py312_2

source /archive/apps/miniconda/miniconda3/py312_2/etc/profile.d/conda.sh
conda activate agrarian312
python "$@"

conda deactivate

# bash <script.py> [--arg1 arg1 --arg2 arg2]

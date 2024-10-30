#! bin/bash

module load proxy/proxy_20
module load cuda12.1

source /archive/apps/miniconda/miniconda3/py310_23.1.0-1/etc/profile.d/conda.sh
conda activate agrarian

python "$@"

conda deactivate

# bash <script.py> [--arg1 arg1 --arg2 arg2]

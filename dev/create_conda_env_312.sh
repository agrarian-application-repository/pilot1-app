#! bin/bash

module load proxy/proxy_16
module load nv-sdk-tool/nvhpc-hpcx-cuda12/23.5
module load miniconda3/py312_2

source /archive/apps/miniconda/miniconda3/py312_2/etc/profile.d/conda.sh
conda create --name agrarian312 python=3.12 -y
conda activate agrarian312

pip install --upgrade pip
pip install --no-cache-dir -r dev/requirements.txt

conda deactivate

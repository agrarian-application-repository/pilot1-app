#! bin/bash

module load proxy/proxy_16
module load cuda12.1
module load miniconda3/py310_23.1.0-1


source /archive/apps/miniconda/miniconda3/py310_23.1.0-1/etc/profile.d/conda.sh
conda create --name agrarian python=3.10 -y
conda activate agrarian

pip install --upgrade pip
pip install --no-cache-dir torch==2.3.1+cu121 torchvision==0.18.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
pip install --no-cache-dir -r requirements.txt

conda deactivate



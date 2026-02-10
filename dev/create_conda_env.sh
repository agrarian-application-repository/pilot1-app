#! bin/bash

conda create --name agrarian312 python=3.12 -y
conda activate agrarian312

pip install --upgrade pip
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install --no-cache-dir -r dev/requirements.txt

conda deactivate

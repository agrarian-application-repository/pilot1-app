#! bin/bash

module load proxy/proxy_20
module load cuda12.1

source /archive/apps/miniconda/miniconda3/py310_23.1.0-1/etc/profile.d/conda.sh
conda activate agrarian

#python finetune.py --data 'src/configs/example_dataset_config_images.yaml' --run_name 'test_run' --epochs 50
python test.py --model 'experiments/test_run/weights/best.pt' --data 'src/configs/example_dataset_config_images.yaml' --split 'test' --run_name 'TEST_test_run'

conda deactivate


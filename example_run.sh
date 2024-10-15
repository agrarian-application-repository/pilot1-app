#! bin/bash

module load proxy/proxy_20
module load cuda12.1

source /archive/apps/miniconda/miniconda3/py310_23.1.0-1/etc/profile.d/conda.sh
conda activate agrarian

#python finetune.py --data 'src/configs/example_dataset_config_images.yaml' --run_name 'test_run' --epochs 50

#python test.py --model 'experiments/test_run/weights/best.pt' --data 'src/configs/example_dataset_config_images.yaml' --split 'test' --run_name 'TEST_test_run'

# IMAGE/IMAGES
#python inference.py --data_path 'data/Aerial_Sheep_v1/test/images/DJI_0004_0262_jpg.rf.14f93587a02be481b46b466ea33580a9.jpg' --checkpoint 'experiments/test_run/weights/best.pt' --output_dir 'INF_IMAGE_test_run'
#python inference.py --data_path 'data/Aerial_Sheep_v1/test/images' --checkpoint 'experiments/test_run/weights/best.pt' --output_dir 'INF_IMAGES_test_run'

#VIDEO/VIDEOS
#python inference.py --data_path 'data/sheep_videos/23.11.10-1.MP4' --checkpoint 'experiments/test_run/weights/best.pt' --output_dir 'INF_VIDEO_test_run' --height 1080 --width 1920
#python inference.py --data_path 'data/sheep_videos_test' --checkpoint 'experiments/test_run/weights/best.pt' --output_dir 'INF_VIDEOS_test_run' --height 1080 --width 1920

python inference.py --data_path 'data/sheep_videos/23.11.10-1.MP4' --checkpoint 'yolo11x.pt' --output_dir 'yolo11x_base' --height 1080 --width 1920

conda deactivate


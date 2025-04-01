#! bin/bash

module load proxy/proxy_20
module load cuda12.1

source /archive/apps/miniconda/miniconda3/py310_23.1.0-1/etc/profile.d/conda.sh
conda activate agrarian

python subsample_video_to_images.py -in '/davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/video1.mp4' -outd '/davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/subsampled' -s 0.1
python subsample_video_to_images.py -in '/davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/video2.mp4' -outd '/davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/subsampled' -s 0.15
python subsample_video_to_images.py -in '/davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/video3.mp4' -outd '/davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/subsampled' -s 0.2

python split_subsampled_images.py --imgdir '/davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/subsampled/video1' --outdir '/davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/split_v1' --train 0.5 --val 0.3
python split_subsampled_images.py --imgdir '/davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/subsampled/video2' --outdir '/davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/split_v1' --train 0.5 --val 0.3
python split_subsampled_images.py --imgdir '/davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/subsampled/video3' --outdir '/davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/split_v1' --train 0.5 --val 0.3

python split_subsampled_images.py --imgdir '/davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/subsampled/video1' --outdir '/davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/split_v2' --train 0.4 --val 0.3
python split_subsampled_images.py --imgdir '/davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/subsampled/video2' --outdir '/davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/split_v2' --train 0.4 --val 0.3
python split_subsampled_images.py --imgdir '/davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/subsampled/video3' --outdir '/davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/split_v2' --train 0.4 --val 0.3

python split_subsampled_images.py --imgdir '/davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/subsampled/video1' --outdir '/davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/split_v3' --train 0.7 --val 0.3
python split_subsampled_images.py --imgdir '/davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/subsampled/video2' --outdir '/davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/split_v3' --train 0.7 --val 0.3
python split_subsampled_images.py --imgdir '/davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/subsampled/video3' --outdir '/davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/split_v3' --train 0.7 --val 0.3

python aggregate_splits.py --input_dir '/davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/split_v1' --output_dir '/davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/aggregated_splits_v1'
python aggregate_splits.py --input_dir '/davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/split_v2' --output_dir '/davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/aggregated_splits_v2'
python aggregate_splits.py --input_dir '/davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/split_v3' --output_dir '/davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/aggregated_splits_v3'
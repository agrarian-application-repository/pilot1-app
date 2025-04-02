#! bin/bash

module load proxy/proxy_20
module load cuda12.1

source /archive/apps/miniconda/miniconda3/py310_23.1.0-1/etc/profile.d/conda.sh
conda activate agrarian

ARCHIVE='/archive/group/ai/datasets/AGRARIAN/MAICH_v2'
ARCHIVE_SUB600='/archive/group/ai/datasets/AGRARIAN/MAICH_v2_SUB600'
ARCHIVE_SUB600_SPLIT='/archive/group/ai/datasets/AGRARIAN/MAICH_v2_SUB600_SPLIT'
ARCHIVE_SUB600_SPLIT_AGGR='/archive/group/ai/datasets/AGRARIAN/MAICH_v2_SUB600_SPLIT_AGGR'

mkdir -p $ARCHIVE_SUB600

# iterate through the MAICH_v2 folder and subforlders to get the names of all MP4 videos
#find "$ARCHIVE" -type f -name "*.MP4" -print0 | while IFS= read -r -d '' file; do
#  echo "$file"
#  python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "$file" -outd "$ARCHIVE_SUB600" -s 15.6
#done

# count number of files
# find /archive/group/ai/datasets/AGRARIAN/MAICH_v2_SUB600 -type f | wc -l

mkdir -p ARCHIVE_SUB600_SPLIT

# Loop through all directories inside ARCHIVE_SUB600
for dir in "$ARCHIVE_SUB600"/*/; do
  # Run the Python command
  echo "Splitting: $dir"
  python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/split_subsampled_images.py --imgdir "$dir" --outdir "$ARCHIVE_SUB600_SPLIT" --train 0.75 --val 0.25
done

mkdir -p ARCHIVE_SUB600_SPLIT_AGGR
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/aggregate_splits.py --input_dir "$ARCHIVE_SUB600_SPLIT" --output_dir "$ARCHIVE_SUB600_SPLIT_AGGR"

# 0.78 - 0.22 actually becomes a 0.815 - 0.195
# 0.75 - 0.25 actually becomes a 0.77 - 0.23
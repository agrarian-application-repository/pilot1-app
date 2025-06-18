#! bin/bash

module load proxy/proxy_20
module load cuda12.1

source /archive/apps/miniconda/miniconda3/py310_23.1.0-1/etc/profile.d/conda.sh
conda activate agrarian

ARCHIVE='/archive/group/ai/datasets/AGRARIAN/MAICH_v2'
ARCHIVE_SUB_AUTO='/archive/group/ai/datasets/AGRARIAN/MAICH_v2_SUB_AUTO'
ARCHIVE_SUB_AUTO_SPLIT='/archive/group/ai/datasets/AGRARIAN/MAICH_v2_SUB_AUTO_SPLIT'
ARCHIVE_SUB_AUTO_SPLIT_AGGR='/archive/group/ai/datasets/AGRARIAN/MAICH_v2_SUB_AUTO_SPLIT_AGGR'

mkdir -p $ARCHIVE_SUB_AUTO

# iterate through the MAICH_v2 folder and subforlders to get the names of all MP4 videos
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202410241029_019/DJI_20241024103403_0001_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 3.0
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202410241029_019/DJI_20241024103617_0002_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 3.0
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202410241029_019/DJI_20241024103846_0003_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 1.5
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202410241029_019/DJI_20241024104019_0004_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 3.0
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202410241029_019/DJI_20241024104237_0005_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 3.0
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202410241029_019/DJI_20241024104502_0006_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 1.5
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202410241029_019/DJI_20241024104715_0007_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 1.5
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202410241029_019/DJI_20241024104935_0008_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 1.5
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202502241159_020/DJI_20250224120046_0001_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 1.5
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202502241159_020/DJI_20250224120324_0002_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 1.5
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202502241159_020/DJI_20250224120558_0003_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 1.5
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202502241159_020/DJI_20250224121245_0004_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 1.5
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202502241159_020/DJI_20250224121827_0005_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 3.0
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202502241223_021/DJI_20250224122443_0001_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 3.0
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202502241223_021/DJI_20250224123005_0002_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 3.0
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202502241223_021/DJI_20250224123534_0003_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 3.0
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202502241223_021/DJI_20250224124109_0004_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 3.0
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202502241223_021/DJI_20250224124755_0005_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 3.0
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202502241340_023/DJI_20250224134208_0001_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 1.5
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202502241340_023/DJI_20250224134619_0002_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 1.5
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202502241357_024/DJI_20250224135834_0001_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 1.5
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202502241357_024/DJI_20250224140006_0002_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 1.5
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202502241357_024/DJI_20250224140114_0003_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 1.5
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202502241357_024/DJI_20250224140224_0004_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 1.5
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202503061245_025/DJI_20250306124741_0001_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 1.5
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202503061323_026/DJI_20250306132455_0001_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 1.5
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202503061323_026/DJI_20250306133155_0002_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 1.5
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202503061323_026/DJI_20250306133227_0003_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 1.5
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202503061323_026/DJI_20250306133927_0004_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 3.0
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202503061341_027/DJI_20250306134200_0001_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 1.5
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202503061341_027/DJI_20250306134445_0002_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 3.0
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202503061341_027/DJI_20250306134844_0003_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 3.0
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202503061341_027/DJI_20250306135545_0004_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 3.0
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202503061341_027/DJI_20250306135816_0005_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 3.0
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202503181150_028/DJI_20250318115157_0001_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 1.5
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202503181150_028/DJI_20250318115602_0002_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 1.5
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202503181150_028/DJI_20250318120303_0003_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 3.0
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202503181150_028/DJI_20250318120631_0004_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 3.0
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202503181150_028/DJI_20250318121331_0005_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 3.0
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202503181216_029/DJI_20250318121836_0001_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 3.0
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202503181216_029/DJI_20250318122237_0002_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 3.0
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202503181216_029/DJI_20250318122512_0003_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 3.0
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202503181216_029/DJI_20250318123128_0004_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 3.0
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202503181216_029/DJI_20250318123658_0005_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 3.0
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/subsample_video_to_images.py -in "/archive/group/ai/datasets/AGRARIAN/MAICH_v2/DJI_202503181242_030/DJI_20250318124342_0001_D.MP4" -outd "$ARCHIVE_SUB_AUTO" -s 1.5


# count number of files
# find /archive/group/ai/datasets/AGRARIAN/MAICH_v2_SUB_AUTO -type f | wc -l

mkdir -p ARCHIVE_SUB_AUTO_SPLIT

# Loop through all directories inside ARCHIVE_SUB_AUTO
for dir in "$ARCHIVE_SUB_AUTO"/*/; do
  # Run the Python command
  echo "Splitting: $dir"
  python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/split_subsampled_images.py --imgdir "$dir" --outdir "$ARCHIVE_SUB_AUTO_SPLIT" --train 0.70 --val 0.15
done

mkdir -p ARCHIVE_SUB_AUTO_SPLIT_AGGR
python /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/aggregate_splits.py --input_dir "$ARCHIVE_SUB_AUTO_SPLIT" --output_dir "$ARCHIVE_SUB_AUTO_SPLIT_AGGR"

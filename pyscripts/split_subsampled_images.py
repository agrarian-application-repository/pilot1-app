import cv2
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
import re
import os
import math
import shutil


def split(input_dir: Path, output_dir: Path, train: float, val: float):
    """
    Orderly split an directory of numbered images in a train, val and test folders

    :param input_dir: Path to the directory of images.
    :param output_dir: Path to the output directory.
    :param train: train split percentage.
    :param val: Validation split percentage.
    """

    # Create train, val, and test directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"

    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(exist_ok=True)

    # Get all files (assuming all are valid numbered .png files)
    image_files = list(input_dir.iterdir())

    # Sort files numerically to ensure orderly split
    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x.stem))))

    # Calculate split indices
    total_images = len(image_files)
    train_size = int(round(total_images * train, 0))
    val_size = int(round(total_images * val, 0))
    test_size = total_images - train_size - val_size

    # Create splits
    train_files = image_files[:train_size]
    val_files = image_files[train_size:train_size + val_size]
    test_files = image_files[train_size + val_size:]

    # Copy files to respective directories with progress bars
    for split_name, files, target_dir in [
        ("train", train_files, train_dir),
        ("validation", val_files, val_dir),
        ("test", test_files, test_dir)
    ]:
        if files:  # Only show progress bar if there are files to copy
            print(f"Copying {split_name} files...")
            for file_path in tqdm(files, desc=f"{split_name.capitalize()} split", unit="file"):
                shutil.copy2(file_path, target_dir / file_path.name)

    # Return information about the split
    split_info = {
        "total": total_images,
        "train": len(train_files),
        "val": len(val_files),
        "test": len(test_files)
    }
    print(split_info)


def check_folder_of_numbered_images(folder_path: Path):
    pattern = re.compile(r"^\d+\.png$")  # Matches filenames like "1.png", "23.png", etc.
    files = list(folder_path.iterdir())
    assert all(file.is_file() for file in files), "Error: Folder contains non-file items."
    assert all(pattern.fullmatch(file.name) for file in files), "Error: Files must be '.png' images named '<Num>.png'"


def main():
    parser = ArgumentParser()
    parser.add_argument("--imgdir", "--image_dir", type=str, required=True)
    parser.add_argument("--outdir", "--output_dir", type=str, required=True)
    parser.add_argument("--train", type=float, required=True)
    parser.add_argument("--val", type=float, required=True)
    args = parser.parse_args()

    # check that the input images dir exists, and that it only contains files named <number>.png
    imgdir = Path(args.imgdir)
    assert imgdir.is_dir(), f"Error: {imgdir} is not a valid path to a directory. Got {args.imgdir}"
    check_folder_of_numbered_images(imgdir)
    
    # check that train and val split are reasonable
    assert 0.0 < args.train < 1.0, f"Error: 'train' must be in (0.0, 1.0). Got {args.train}"
    assert 0.0 < args.val < 1.0, f"Error: 'val' must be in (0.0, 1.0). Got {args.val}"
    assert 0.0 < args.train + args.val <= 1.0, f"Error: 'train'+'val' must be in (0.0, 1.0]. Got {args.train + args.val}"
    args.test = 1.0 - (args.train + args.val)

    # create the output directory from the provided path and the name of the image dir + train/val/test split percentages
    output_dir = Path(args.outdir, f"{imgdir.stem}_train{args.train*100:.0f}_val{args.val*100:.0f}_test{args.test*100:.0f}")
    output_dir.mkdir(parents=True, exist_ok=True)

    split(
        input_dir=imgdir,
        output_dir=output_dir,
        train=args.train,
        val=args.val,
    )


if __name__ == "__main__":
    main()


import re
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm


def validate_directory_structure(input_dir):
    """
    Validates that the input directory contains only directories with the
    naming pattern <name>_train<train_perc>_val<val_perc>_test<test_perc>
    and checks that all percentages are consistent.
    """
    input_path = Path(input_dir)

    if not input_path.exists() or not input_path.is_dir():
        raise ValueError(f"Input directory {input_dir} does not exist or is not a directory")

    # Get all subdirectories
    subdirs = [d for d in input_path.iterdir() if d.is_dir()]

    if not subdirs:
        raise ValueError(f"Input directory {input_dir} does not contain any subdirectories")

    # Regular expression to extract percentages from directory names
    pattern = r".*_train(\d+)_val(\d+)_test(\d+)$"

    # Extract percentages from first directory to compare with others
    reference_dir = subdirs[0].name
    match = re.match(pattern, reference_dir)

    if not match:
        raise ValueError(f"Directory {reference_dir} does not follow the required naming pattern")

    reference_train, reference_val, reference_test = match.groups()

    # Validate all other directories
    valid_dirs = []
    for subdir in subdirs:
        match = re.match(pattern, subdir.name)

        if not match:
            print(f"Warning: Directory {subdir.name} does not follow the required naming pattern and will be skipped")
            continue

        train_perc, val_perc, test_perc = match.groups()

        if (train_perc != reference_train or
                val_perc != reference_val or
                test_perc != reference_test):
            print(f"Warning: Directory {subdir.name} has different percentages and will be skipped")
            continue

        # Check if it contains train, val, and test subdirectories
        if not ((subdir / "train").exists() and (subdir / "val").exists() and (subdir / "test").exists()):
            print(f"Warning: Directory {subdir.name} is missing train, val, or test subdirectories and will be skipped")
            continue

        valid_dirs.append(subdir)

    if not valid_dirs:
        raise ValueError("No valid directories found that meet all requirements")

    return valid_dirs, (reference_train, reference_val, reference_test)


def merge_datasets(valid_dirs, output_dir):
    """
    Merges the train, val, and test datasets from valid directories
    into a single output directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create output train, val, and test directories
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    test_dir = output_path / "test"

    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(exist_ok=True)

    # Copy files from each valid directory to the output directory
    for subdir in valid_dirs:
        subdir_name = subdir.name.split('_')[0]  # Extract the <name> part

        for split in ["train", "val", "test"]:
            src_dir = subdir / split
            dst_dir = output_path / split

            # Get all files in the source directory
            files = list(src_dir.iterdir())

            if not files:
                print(f"Warning: {split} directory in {subdir.name} is empty")
                continue

            print(f"Copying {len(files)} files from {subdir.name}/{split} to output/{split}...")

            # Copy files with a prefix to avoid name conflicts
            for file_path in tqdm(files, desc=f"{subdir_name} {split}", unit="file"):
                if file_path.is_file():
                    # Add the subdirectory name as a prefix to avoid name conflicts
                    new_name = f"{subdir_name}_{file_path.name}"
                    shutil.copy2(file_path, dst_dir / new_name)


def main():
    parser = argparse.ArgumentParser(description="Merge train/val/test datasets from multiple directories")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory containing subdirectories with splits")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory containing aggregated splits")
    args = parser.parse_args()

    print(f"Validating directory structure in {args.input_dir}...")
    valid_dirs, percentages = validate_directory_structure(args.input_dir)

    train_perc, val_perc, test_perc = percentages
    print(f"Found {len(valid_dirs)} valid directories with train {train_perc}%, val {val_perc}%, test {test_perc}%")

    print(f"Merging datasets into {args.output_dir}...")
    merge_datasets(valid_dirs, args.output_dir)

    print("Dataset merge completed successfully!")


if __name__ == "__main__":
    main()

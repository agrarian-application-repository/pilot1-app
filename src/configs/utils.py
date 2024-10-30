import argparse
import re
from argparse import Namespace
from pathlib import Path
from typing import Any, Optional, Union

import yaml

ALLOWED_IMAGE_FORMATS = [
    ".bmp",
    ".dng",
    ".jpeg",
    ".jpg",
    ".mpo",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
    ".pfm",
]
ALLOWED_VIDEO_FORMATS = [
    ".asf",
    ".avi",
    ".gif",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".ts",
    ".wmv",
    ".webm",
]

BASE_CHECKPOINTS = [
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt",
]
BASE_TRACKERS = ["botsort.yaml", "bytetrack.yaml"]


def parse_config_file() -> str:
    """
    Parses the command line arguments to retrieve the path to the YAML configuration file.

    :return: Path to the YAML configuration file.
    """
    parser = argparse.ArgumentParser(
        description="Parse YAML configuration file for running the experiment."
    )

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file."
    )

    args = parser.parse_args()

    return args.config


def parse_detect_segment_config_files() -> tuple[str, str]:
    """
    Parses the command line arguments to retrieve the path to the YAML configuration file.

    :return: Path to the detector YAML configuration file.
    :return: Path to the segmenter YAML configuration file.
    """
    parser = argparse.ArgumentParser(
        description="Parse detector and segmenter YAML configuration files for running the experiment."
    )

    parser.add_argument(
        "--det_config",
        type=str,
        required=True,
        help="Path to the detector YAML configuration file.",
    )
    parser.add_argument(
        "--seg_config",
        type=str,
        required=True,
        help="Path to the segmenter YAML configuration file.",
    )

    args = parser.parse_args()

    return args.det_config, args.seg_config


def read_yaml_config(yaml_file: str) -> dict[str, Any]:
    """
    Reads the training configuration from a YAML file.

    :param yaml_file: Path to the YAML file.
    :return: Dictionary of the parsed YAML configuration.
    """
    try:
        with open(yaml_file, "r") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: YAML configuration file '{yaml_file}' not found.")
        exit()
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
        exit()


def is_youtube_link(url: str) -> bool:
    """
    Check if the given URL is a valid YouTube link.

    Args:
        url (str): The URL string to be checked.

    Returns:
        bool: True if the URL is a valid YouTube link, otherwise False.
    """
    # Regular expression pattern for YouTube URLs
    youtube_regex = re.compile(
        r"^(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/"
        r"(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})"
    )

    # Match the URL against the pattern
    match = youtube_regex.match(url)

    return bool(match)


def is_valid_image(file: Path) -> bool:
    """
    Check if the given file is a valid image.

    Args:
        file (Path): The file path to be checked.

    Returns:
        bool: True if the file has a valid image file extension, otherwise False.
    """
    return file.suffix.lower() in ALLOWED_IMAGE_FORMATS


def is_valid_list_images(files: list[Path]) -> bool:
    """
    Checks whether the provided list of files contains only images in allowed format.

    Args:
        files (List[Path]): List of file paths in the directory.

    Returns:
        bool: True if the list is not empty and it contains image files in allowed format, otherwise False.
    """
    return len(files) > 0 and all(
        file.suffix.lower() in ALLOWED_IMAGE_FORMATS for file in files
    )


def is_valid_video(file: Path) -> bool:
    """
    Check if the given file is a valid video.

    Args:
        file (Path): The file path to be checked.

    Returns:
        bool: True if the file has a valid video file extension, otherwise False.
    """
    return file.suffix.lower() in ALLOWED_VIDEO_FORMATS


def is_valid_list_videos(files: list[Path]) -> bool:
    """
    Checks whether the provided list of files contains only videos in allowed format.

    Args:
        files (List[Path]): List of file paths in the directory.

    Returns:
        bool: True if the list is not empty and it contains video files in allowed format, otherwise False.
    """
    return len(files) > 0 and all(
        file.suffix.lower() in ALLOWED_VIDEO_FORMATS for file in files
    )


def is_valid_checkpoint(checkpoint: Path) -> bool:
    """
    Check if the given file path is a valid checkpoint file, i.e. if it exists and is a .pt file.

    Args:
        checkpoint (Path): The file path to be checked.

    Returns:
        bool: True if the file exists and has a `.pt` extension, otherwise False.
    """
    return is_valid_pt_pt_file(checkpoint) or str(checkpoint) in BASE_CHECKPOINTS


def is_valid_tracker(conf_file: Path) -> bool:
    return is_valid_yaml_conf(conf_file) or str(conf_file) in BASE_TRACKERS


def is_valid_pt_pt_file(pt_file: Path) -> bool:
    """
    Check if the given file path is a valid checkpoint file, i.e. if it exists and is a .pt file.

    Args:
        pt_file (Path): The file path to be checked.

    Returns:
        bool: True if the file exists and has a `.pt` extension, otherwise False.
    """
    return pt_file.exists() and pt_file.is_file() and pt_file.suffix == ".pt"


def is_valid_yaml_conf(conf_file: Path) -> bool:
    """
    Check if the given file path is a valid yaml file, i.e. if it exists and is a .yaml file.

    Args:
        conf_file (Path): The file path to be checked.

    Returns:
        bool: True if the file exists and has a `.yaml` extension, otherwise False.
    """
    return conf_file.exists() and conf_file.is_file() and conf_file.suffix == ".yaml"


def get_arguments_dict(args: Namespace) -> dict[str, Any]:
    return vars(args)


def check_bounded(
    value: Union[int, float],
    lower: Optional[Union[int, float]] = None,
    upper: Optional[Union[int, float]] = None,
    strict_lower: bool = False,
    strict_upper: bool = False,
) -> None:
    """
    Check if a value is within specified bounds (lower, upper). The bounds can be strict or non-strict.

    Parameters:
    - value: The value to check (int or float).
    - lower: The lower bound (optional, int or float). If None, no lower bound is applied.
    - upper: The upper bound (optional, int or float). If None, no upper bound is applied.
    - strict_lower: Whether the lower bound should be strict (value must be greater than lower).
    - strict_upper: Whether the upper bound should be strict (value must be less than upper).

    Returns:
    - None if the value is within bounds.

    Raises:
    - ValueError if the value violates any bounds, with meaningful error messages.
    """

    # Check for lower bound
    if lower is not None:
        if strict_lower:
            if value <= lower:
                raise ValueError(
                    f"Value {value} must be strictly greater than the lower bound {lower}."
                )
        else:
            if value < lower:
                raise ValueError(
                    f"Value {value} must be greater than or equal to the lower bound {lower}."
                )

    # Check for upper bound
    if upper is not None:
        if strict_upper:
            if value >= upper:
                raise ValueError(
                    f"Value {value} must be strictly less than the upper bound {upper}."
                )
        else:
            if value > upper:
                raise ValueError(
                    f"Value {value} must be less than or equal to the upper bound {upper}."
                )

    # If the function reaches this point, the value is valid
    return None

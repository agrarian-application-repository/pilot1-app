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

BASE_CHECKPOINTS_DETECT = [
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt",
]

BASE_CHECKPOINTS_SEGMENT = [
    "yolo11n-seg.pt",
    "yolo11s-seg.pt",
    "yolo11m-seg.pt",
    "yolo11l-seg.pt",
    "yolo11x-seg.pt",
]

BASE_TRACKERS = ["botsort.yaml", "bytetrack.yaml"]


def parse_config_file() -> str:
    """
    Parses the command line arguments to retrieve the path to the YAML configuration file.

    :return: Path to the YAML configuration file.
    """
    parser = argparse.ArgumentParser(description="Parse YAML configuration file for running the experiment.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    return args.config


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


def is_valid_pt_file(pt_file: Path) -> bool:
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


def is_valid_checkpoint(checkpoint: Path, task: str) -> bool:
    """
    Check if the given file path is a valid checkpoint file, i.e. if it exists and is a .pt file.

    Args:
        checkpoint (Path): The file path to be checked.
        task (str): The task to perform.

    Returns:
        bool: True if the file exists and has a `.pt` extension, otherwise False.
    """
    if task == "detect":
        return (str(checkpoint) in BASE_CHECKPOINTS_DETECT) or is_valid_pt_file(checkpoint)
    elif task == "segment":
        return (str(checkpoint) in BASE_CHECKPOINTS_SEGMENT) or is_valid_pt_file(checkpoint)
    else:
        raise NotImplementedError(f"Model check for task {task} not implemented")


def is_valid_tracker(conf_file: Path) -> bool:
    return is_valid_yaml_conf(conf_file) or str(conf_file) in BASE_TRACKERS


def is_valid_youtube_link(url: str) -> bool:
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


def is_valid_image(img_path: Path) -> bool:
    """
    Check if the given img_path corresponds to a valid image.

    Args:
        img_path (Path): The image path to be checked.

    Returns:
        bool: True if the path corresponds to a valid image, otherwise False.
    """
    return img_path.exists() and img_path.is_file() and img_path.suffix.lower() in ALLOWED_IMAGE_FORMATS


def is_valid_images_dir(dir_path: Path) -> bool:
    """
    Checks whether the provided path corresponds to a directory containing only valid images.

    Args:
        dir_path (Path): Path to the directory.

    Returns:
        bool: True if the provided path corresponds to a directory containing only valid images, otherwise False.
    """

    if not dir_path.exists() or not dir_path.is_dir():
        return False

    for item in dir_path.iterdir():
        if not is_valid_image(item):
            return False

    return True


def is_valid_video(video_path: Path) -> bool:
    """
    Check if the given video_path corresponds to a valid video.

    Args:
        video_path (Path): The video path to be checked.

    Returns:
        bool: True if the path corresponds to a valid video, otherwise False.
    """
    return video_path.exists() and video_path.is_file() and video_path.suffix.lower() in ALLOWED_VIDEO_FORMATS


def is_valid_videos_dir(dir_path: Path) -> bool:
    """
    Checks whether the provided path corresponds to a directory containing only valid videos.

    Args:
        dir_path (Path): Path to the directory.

    Returns:
        bool: True if the provided path corresponds to a directory containing only valid videos, otherwise False.
    """

    if not dir_path.exists() or not dir_path.is_dir():
        return False

    for item in dir_path.iterdir():
        if not is_valid_video(item):
            return False

    return True






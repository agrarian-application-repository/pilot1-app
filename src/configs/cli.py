import re
from pathlib import Path

ALLOWED_IMAGE_FORMATS = [".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm"]
ALLOWED_VIDEO_FORMATS = [".asf", ".avi", ".gif", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".ts", ".wmv",
                         ".webm"]


def is_float_in_01(metric: float) -> bool:
    """
    Check if a given value "metric" is a float within the range [0, 1].

    Args:
        metric (float): The value to be checked.

    Returns:
        bool: True if the "metric" value is a float within [0, 1], otherwise False.
    """
    return 0 <= metric <= 1


def is_positive_integer(value: int) -> bool:
    """
    Check if a given value is a positive integer.

    Args:
        value (int): The value to be checked.

    Returns:
        bool: True if the value is a positive integer, otherwise False.
    """
    return value > 0


def is_valid_checkpoint(checkpoint: Path) -> bool:
    """
    Check if the given file path is a valid checkpoint file, i.e. if it exists and is a .pt file.

    Args:
        checkpoint (Path): The file path to be checked.

    Returns:
        bool: True if the file exists and has a `.pt` extension, otherwise False.
    """
    return checkpoint.exists() and checkpoint.suffix == ".pt"


def is_valid_yaml_conf(conf_file: Path) -> bool:
    """
    Check if the given file path is a valid yaml file, i.e. if it exists and is a .yaml file.

    Args:
        conf_file (Path): The file path to be checked.

    Returns:
        bool: True if the file exists and has a `.yaml` extension, otherwise False.
    """
    return conf_file.exists() and conf_file.suffix == ".yaml"


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
        r'^(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
        r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
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
    return len(files) > 0 and all(file.suffix.lower() in ALLOWED_IMAGE_FORMATS for file in files)


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
    return len(files) > 0 and all(file.suffix.lower() in ALLOWED_VIDEO_FORMATS for file in files)


# TODO DOCSTRINGS
def check_inference_arguments(args) -> str:
    # Check confidence and IOU values are within [0, 1]
    if not is_float_in_01(args.conf):
        print(f"ERROR: confidence must be in [0,1]. Got {args.conf}")
        exit()

    if not is_float_in_01(args.iou):
        print(f"ERROR: iou must be in [0,1]. Got {args.iou}")
        exit()

    # Check height, width and batch_size are > 0
    if not is_positive_integer(args.height):
        print(f"ERROR: height must be a positive integer. Got {args.height}")
        exit()

    if not is_positive_integer(args.width):
        print(f"ERROR: width must be a positive integer. Got {args.width}")
        exit()

    if not is_positive_integer(args.batch_size):
        print(f"ERROR: batch_size must be a positive integer. Got {args.batch_size}")
        exit()

    # Check if the model checkpoint is valid
    model_checkpoint = Path(args.checkpoint)
    if not is_valid_checkpoint(model_checkpoint):
        print(f"ERROR: model checkpoint {model_checkpoint} was not found.")
        exit()

    # Begin checking the data_path ...
    data_path = args.data_path

    # Check if the data path is a YouTube link
    if is_youtube_link(data_path):
        return "youtube"

    else:
        data_path = Path(data_path)

        # If the data path is a directory, check if it contains only valid images or videos
        if data_path.is_dir():
            files = [file for file in data_path.iterdir()]

            if is_valid_list_images(files):
                return "images"
            if is_valid_list_videos(files):
                return "videos"

        # If the data path is a single file, check if it is a valid image or video
        elif data_path.is_file():
            if is_valid_image(data_path):
                return "image"
            if is_valid_video(data_path):
                return "video"

    # If none of the checks pass, raise an error and exit
    print(f"ERROR: ({data_path}) is not an image, a folder of images, a video, nor a youtube link")
    exit()


# TODO DOCSTRINGS
def check_test_arguments(args):
    # Check confidence and IOU values are within [0, 1]
    if not is_float_in_01(args.conf):
        print(f"ERROR: confidence must be in [0,1]. Got {args.conf}")
        exit()

    if not is_float_in_01(args.iou):
        print(f"ERROR: iou must be in [0,1]. Got {args.iou}")
        exit()

    # Check height, width and batch_size are > 0
    if not is_positive_integer(args.imgsz):
        print(f"ERROR: imgsz must be a positive integer. Got {args.imgsz}")
        exit()

    if not is_positive_integer(args.batch_size):
        print(f"ERROR: batch_size must be a positive integer. Got {args.batch_size}")
        exit()

    # Check if the model checkpoint is valid
    model_checkpoint = Path(args.checkpoint)
    if not is_valid_checkpoint(model_checkpoint):
        print(f"ERROR: model checkpoint {model_checkpoint} was not found.")
        exit()

    conf_file = Path(args.data)
    if not is_valid_yaml_conf(conf_file):
        print(f"ERROR: data config file {conf_file} was not found.")
        exit()



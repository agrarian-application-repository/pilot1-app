import os
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Union

from dotenv import load_dotenv
from ultralytics import YOLO

import wandb

ALLOWED_IMAGE_FORMATS = [".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm"]
ALLOWED_VIDEO_FORMATS = [".asf", ".avi", ".gif", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".ts", ".wmv",
                         ".webm"]


def parse_cli_arguments():
    parser = ArgumentParser()

    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)

    parser.add_argument('--height', type=int, default=640, help="Image/video height for training (default: 640)")
    parser.add_argument('--width', type=int, default=640, help="Image/video width for training (default: 640)")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for testing (default: 16)")
    parser.add_argument('--conf_thresh', type=float, default=0.25,
                        help="Confidence threshold for detecting objects (default: 0.25)")
    parser.add_argument('--iou_thresh', type=float, default=0.70,
                        help="IoU threshold for NMS (Non-Max Suppression) (default: 0.70)")

    parser.add_argument("--run_name", type=str, required=True)

    args = parser.parse_args()

    return args


def is_in_01(metric: float):
    return 0 <= metric <= 1


def is_valid_checkpoint(checkpoint: Path):
    return checkpoint.exists() and checkpoint.suffix == ".pt"


def is_youtube_link(url: str):
    # Regular expression pattern for YouTube URLs
    youtube_regex = re.compile(
        r'^(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
        r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')

    # Match the URL against the pattern
    match = youtube_regex.match(url)

    return bool(match)


def is_valid_image(file: Path):
    return file.suffix.lower() in ALLOWED_VIDEO_FORMATS


def is_valid_video(file: Path):
    return file.suffix.lower() in ALLOWED_VIDEO_FORMATS


def is_valid_images_dir(files: list[Path]):
    return len(files) > 0 and all(file.suffix.lower() in ALLOWED_IMAGE_FORMATS for file in files)


def is_valid_videos_dir(files: list[Path]):
    return len(files) > 0 and all(file.suffix.lower() in ALLOWED_VIDEO_FORMATS for file in files)


def check_cli_arguments(args):
    if not is_in_01(args.conf):
        print(f"ERROR: confidence must be in [0,1]. Got {args.conf}")
        exit()

    if not is_in_01(args.iou):
        print(f"ERROR: iou must be in [0,1]. Got {args.iou}")
        exit()

    model_checkpoint = Path(args.checkpoint)
    if not is_valid_checkpoint(model_checkpoint):
        print(f"ERROR: model checkpoint {args.checkpoint} was not found.")
        exit()

    data_path = args.data_path

    if is_youtube_link(data_path):
        return "youtube"

    else:
        data_path = Path(data_path)

        if args.data_path.isdir():

            files = [file for file in data_path.iterdir() if file.is_file()]

            if is_valid_images_dir(files):
                return "images"
            if is_valid_videos_dir(files):
                return "videos"

        elif args.data_path.isfile():
            if is_valid_image(data_path):
                return "image"
            if is_valid_video(data_path):
                return "video"

    print(f"ERROR: ({data_path}) is not an image, video, folder containing them, nor a youtube link")
    exit()


def run_inference(
        model,
        media_type: str,
        data_path: Union[Path, str],
        output_dir: Path,
        imgsz: tuple[float, float],
        conf: float,
        iou: float,
        device: str,
):
    """
    Run YOLO11 inference on the given input source (image, video, or YouTube URL).

    :param imgsz:
    :param model:
    :param media_type:
    :param data_path:
    :param output_dir:
    :param conf:
    :param iou:
    :param device:
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    if media_type in ["images", "videos", "youtube"]:
        source = [file for file in data_path.iterdir() if file.is_file()]
        stream = True
    else:
        source = data_path
        stream = False

    model.predict(
        source=source,
        classes=[0],
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        stream=stream,
        save=True,
        save_dir=output_dir,
    )


def main():
    # Load environment variables from .env file
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")

    # Ensure W&B API key is loaded
    if wandb_api_key is None:
        raise ValueError("W&B API key not found. Please check your .env file.")

    args = parse_cli_arguments()

    wandb.login(key=wandb_api_key)
    wandb.init(
        project="experiments",
        name=args.run_name,
    )

    media_type = check_cli_arguments(args)
    model_checkpoint = args.checkpoint
    data_path = Path(args.data_path) if media_type != "youtube" else args.data_path
    output_dir = Path(args.output_dir)
    imgsz = (args.height, args.width)
    conf = args.conf
    iou = args.iou
    device = args.device

    model = YOLO(model_checkpoint)

    run_inference(model, media_type, data_path, output_dir, imgsz, conf, iou, device)

    # Finish wandb logging
    wandb.finish()


if __name__ == "__main__":
    main()

from pathlib import Path
from typing import Any

import cv2
from src.config_v2.utils import (
    is_valid_image,
    is_valid_list_images,
    is_valid_list_videos,
    is_valid_video,
    is_youtube_link,
)


def check_inference_args(args: dict[str, Any]) -> dict[str, Any]:
    data_path = args["source"]

    # Check if the data path is a YouTube link
    if is_youtube_link(data_path):
        args["_media"] = "youtube"

    else:
        data_path = Path(data_path)

        # If the data path is a directory, check if it contains only valid images or videos
        if data_path.is_dir():
            files = [file for file in data_path.iterdir()]

            if is_valid_list_images(files):
                args["_media"] = "images"
            elif is_valid_list_videos(files):
                args["_media"] = "videos"
            else:
                print(
                    f"ERROR: ({data_path}) is not a folder of valid images nor videos"
                )
                exit()

        # If the data path is a single file, check if it is a valid image or video
        elif data_path.is_file():
            if is_valid_image(data_path):
                args["_media"] = "image"
            elif is_valid_video(data_path):
                args["_media"] = "video"
            else:
                print(f"ERROR: ({data_path}) is not a valid image or video")
                exit()

        # If none of the checks pass, raise an error and exit
        print(
            f"ERROR: ({data_path}) is not a valid image, video, folder of images or video, nor a youtube link"
        )
        exit()

    return args


def preprocess_inference_args(args: dict[str, Any]) -> dict[str, Any]:
    media_type = args.pop("media")

    # Transform source paths, if necessary
    if media_type != "youtube":
        args["source"] = Path(args["source"])
    else:
        pass  # Leave youtube path as is

    # assign "stream" attribute based on media type
    if media_type == "image":
        args["stream"] = False
    else:  # media_type in ["images", "video", "videos", "youtube"]
        args["stream"] = True

    # Handle dimensions
    if args["width"] is None or args["height"] is None:

        if media_type == "image":
            height, width, _ = cv2.imread(args["source"]).shape
        elif media_type == "images":
            first_image = [
                subpath for subpath in args["source"].iterdir() if subpath.is_file()
            ][0]
            height, width, _ = cv2.imread(first_image).shape
        elif media_type == "video":
            video = cv2.VideoCapture(args["source"])
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        elif media_type == "videos":
            first_videos = [
                subpath for subpath in args["source"].iterdir() if subpath.is_file()
            ][0]
            height = int(first_videos.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(first_videos.get(cv2.CAP_PROP_FRAME_WIDTH))
        else:  # youtube
            yt_video = cv2.VideoCapture(args["source"])
            height = int(yt_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(yt_video.get(cv2.CAP_PROP_FRAME_WIDTH))

    args["width"] = args["width"] if args["width"] is not None else width
    args["height"] = args["height"] if args["height"] is not None else height

    return args

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO, solutions

from src.configs.cli import BASE_CHECKPOINTS, check_tracking_arguments


def parse_cli_arguments():
    parser = ArgumentParser()

    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to a video or a youtube link"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a pretrained YOLO model checkpoint (.pt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path where to save the tracking results within ./experiments",
    )
    parser.add_argument(
        "--output_name", type=str, required=True, help="Name of the output video file"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Video frame height (default: original height)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Video frame width (default: original width)",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detecting objects (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.70,
        help="IoU threshold for NMS (Non-Max Suppression) (default: 0.70)",
    )

    parser.add_argument(
        "--count", type=int, default=0, help="Whether to display objects count"
    )
    parser.add_argument(
        "--movement",
        type=int,
        default=0,
        help="Whether to draw lines to track objects movements",
    )
    parser.add_argument(
        "--heatmap",
        type=int,
        default=0,
        help="Whether to overlap heatmap over background",
    )
    parser.add_argument(
        "--distance",
        type=int,
        default=0,
        help="Whether to show distances between objects",
    )
    parser.add_argument(
        "--speed", type=int, default=0, help="Whether to show speed of objects"
    )

    parser.add_argument(
        "--tracker_path",
        type=str,
        default="botsort.yaml",
        help="Path to a tracker configuration file (.yaml)",
    )
    # https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers

    # TODO FIX DEVICE can be string, list, int
    parser.add_argument(
        "--device",
        type=int,
        default=3,
        help="device where the model is run, can be 'cpu' a number for the corresponding gpu, "
        "for a list of numbers for distributed inference",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_cli_arguments()

    model = YOLO(args.checkpoint)

    if args.checkpoint in BASE_CHECKPOINTS:
        classes = [18]  # Pretrained "sheep" class id
    else:
        classes = [0]  # Finetuned "sheep" class id

    video_path = check_cv2_video_tracking_arguments(args)

    results = model.track(
        tracker=args.tracker_path,
        persist=True,
        source=data_path,
        classes=[0],
        imgsz=(args.height, args.width),
        conf=args.conf,
        iou=args.iou,
        stream=True,
        save=True,
        device=args.device,
    )

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"

    # Get the width (W), height (H), and frames per second (FPS) of the input video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # if width or height are input arguments, overwrite the corresponding dimensions for the output video
    frame_width = frame_width if args.width is None else args.width
    frame_height = frame_height if args.height is None else args.width

    # define the output video path
    output_file_name = args.output_name + ".mp4"
    output_path = str(Path("../experiments", args.output_dir, output_file_name))

    # Define the codec and create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 files
    video_writer = cv2.VideoWriter(
        output_path, fourcc, fps, (frame_width, frame_height)
    )

    # Store the track history
    track_history = defaultdict(lambda: [])

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO11 tracking on the frame, persisting tracks between frames
            results = model.track(
                source=frame,
                classes=classes,
                imgsz=(frame_height, frame_width),
                conf=args.conf,
                iou=args.iou,
                tracker="botsort.yaml",
                persist=True,
                device=args.device,
            )

            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    annotated_frame,
                    [points],
                    isClosed=False,
                    color=(230, 230, 230),
                    thickness=10,
                )

            # Write the annotated frame to the output video file
            video_writer.write(annotated_frame)

        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object, video writer, and close any opened windows
    cap.release()
    video_writer.release()
    # cv2.destroyAllWindows()

    # TODO log results
    """
    wandb_api_key = get_wandb_api_key()

    wandb.login(key=wandb_api_key)
    wandb.init(
        project="agrarian",
        name="test_inference",
    )

    # Finish wandb logging
    wandb.finish()
    """


if __name__ == "__main__":
    main()

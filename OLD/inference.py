from argparse import ArgumentParser
from pathlib import Path

from ultralytics import YOLO

from src.configs.cli import check_inference_arguments


def parse_cli_arguments():
    parser = ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to an image, a video, a folder containing images or videos, or a youtube link",
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
        help="Path where to save the inference results within ./experiments",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=640,
        help="Image/video height for training (default: 640)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Image/video width for training (default: 640)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for testing (default: 16)",
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

    media_type = check_inference_arguments(args)

    data_path = Path(args.data_path) if media_type != "youtube" else args.data_path

    if media_type in ["image"]:
        stream = False
    else:  # media_type in ["images", "video", "videos", "youtube"]
        stream = True

    model = YOLO(args.checkpoint)

    results = model.predict(
        source=data_path,
        classes=[0],
        imgsz=(args.height, args.width),
        conf=args.conf,
        iou=args.iou,
        stream=stream,
        save=True,
        project="experiments",
        name=args.output_dir,
        device=args.device,
    )

    if stream:
        print("saving with stream")
        # iterate through generator to trigger saving of results
        for r in results:
            pass

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

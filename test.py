import argparse
from src.logging.utils import get_wandb_api_key
from ultralytics import YOLO
from pathlib import Path

import wandb


# Parse command-line arguments using argparse
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, required=True,
                        help="Path to the dataset YAML configuration file")
    parser.add_argument('--split', type=str, default="test", choices=["train", "val", "test"],
                        help="dataset split on which to validate the model (default: test)")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a pretrained YOLO model checkpoint (.pt)")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path where to save the inference results within ./experiments")

    parser.add_argument('--imgsz', type=int, default=640,
                        help="Image/video size for training (default: 640)")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size for testing (default: 16)")

    parser.add_argument('--conf', type=float, default=0.25,
                        help="Confidence threshold for detecting objects (default: 0.25)")
    parser.add_argument('--iou', type=float, default=0.70,
                        help="IoU threshold for NMS (Non-Max Suppression) (default: 0.70)")

    # TODO FIX DEVICE can be string, list, int
    parser.add_argument("--device", type=int, default=3,
                        help="device where the model is run, can be 'cpu' a number for the corresponding gpu, "
                             "for a list of numbers for distributed inference")

    return parser.parse_args()


def main():
    # Parse the command-line arguments
    args = parse_args()

    # Load the trained YOLOv11 model
    model = YOLO(args.checkpoint)

    # Perform validation (which includes evaluation on the test set)
    results = model.val(
        data=Path(args.data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch_size,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
    )

    # TODO log results
    """
    wandb_api_key = get_wandb_api_key()

    # Initialize W&B logging
    wandb.login(key=wandb_api_key)
    wandb.init(
        project="agrarian",
        name="test_test"
    )

    # Log test parameters to W&B
    wandb.config.update({
        "model": args.model,
        "data": args.data,
        "split": args.split,
        "imgsz": args.imgsz,
        "batch_size": args.batch_size,
        "conf": args.conf,
        "iou": args.iou,
    })

    # Log validation metrics to W&B
    wandb.log(dict(results.results_dict))
    wandb.log({
        "box_mAP50-95": results.box.map,  # mAP50-95
        "box_mAP50": results.box.map50,  # mAP50
        "box_mAP75": results.box.map75,  # mAP75
        "box_cat_mAP50-95": results.box.maps,  # list of mAP50-95 for each category)
    })

    # Finish the W&B run
    wandb.finish()
    """


if __name__ == "__main__":
    main()

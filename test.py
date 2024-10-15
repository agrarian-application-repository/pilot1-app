import argparse
import os

from dotenv import load_dotenv
from ultralytics import YOLO

import wandb


# Parse command-line arguments using argparse
def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Testing and Evaluation with W&B")

    # Model and dataset paths
    parser.add_argument('--model', type=str, required=True,
                        help="Path to the trained YOLO11 model or checkpoint (e.g., best.pt)")
    parser.add_argument('--data', type=str, required=True, help="Path to the dataset YAML file (for testing dataset)")
    parser.add_argument('--split', type=str, default="test", choices=["train", "val", "test"],
                        help="Path to the dataset YAML file (for testing dataset)")
    parser.add_argument('--run_name', type=str, default="yolo11_test",
                        help="Name of the test run in W&B (default: yolo11_test)")

    # Test-specific parameters
    parser.add_argument('--imgsz', type=int, default=640, help="Image/video size for training (default: 640)")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for testing (default: 16)")
    parser.add_argument('--conf_thresh', type=float, default=0.25,
                        help="Confidence threshold for detecting objects (default: 0.25)")
    parser.add_argument('--iou_thresh', type=float, default=0.70,
                        help="IoU threshold for NMS (Non-Max Suppression) (default: 0.70)")

    parser.add_argument('--device', type=str, default="3", help="device to use for training (default 3)")

    return parser.parse_args()


def main():

    # Load environment variables from .env file
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")

    # Ensure W&B API key is loaded
    if wandb_api_key is None:
        raise ValueError("W&B API key not found. Please check your .env file.")

    # Parse the command-line arguments
    args = parse_args()

    # Initialize W&B logging
    wandb.login(key=wandb_api_key)
    wandb.init(
        project="experiments",
        name=args.run_name
    )

    # Log test parameters to W&B
    wandb.config.update({
        "model": args.model,
        "data": args.data,
        "split": args.split,
        "imgsz": args.imgsz,
        "batch_size": args.batch_size,
        "conf_thresh": args.conf_thresh,
        "iou_thresh": args.iou_thresh,
    })

    # Load the trained YOLOv8 model
    model = YOLO(args.model)

    # Perform validation (which includes evaluation on the test set)
    results = model.val(
        data=args.data,  # Path to the dataset YAML file
        split=args.split,
        imgsz=args.imgsz,  # Image size for evaluation
        batch=args.batch_size,  # Batch size for evaluation
        conf=args.conf_thresh,  # Confidence threshold for detection
        iou=args.iou_thresh,  # IoU threshold for NMS
        device=args.device,
        verbose=False,  # Print detailed logs during evaluation
    )

    # Log validation metrics to W&B
    wandb.log(dict(results.results_dict))
    wandb.log({
        "box_mAP50-95": results.box.map, # mAP50-95
        "box_mAP50": results.box.map50,  # mAP50
        "box_mAP75": results.box.map75,  # mAP75
        "box_cat_mAP50-95": results.box.maps,  # list of mAP50-95 for each category)
    })

    # Finish the W&B run
    wandb.finish()

    print(results)


if __name__ == "__main__":
    main()

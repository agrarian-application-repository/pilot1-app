import argparse
import os

from dotenv import load_dotenv
from ultralytics import YOLO

import wandb


# Parse command-line arguments using argparse
def parse_args():
    parser = argparse.ArgumentParser(description="YOLO11 Fine-tuning with W&B")

    # Dataset and model parameters
    parser.add_argument('--data', type=str, required=True, help="Path to the dataset YAML file")
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                        help="Pretrained YOLO11 model path or name (default: yolo11.pt)")
    parser.add_argument('--run_name', type=str, default="yolo11_finetune",
                        help="Name of the training run (default: yolo11_finetune)")

    # Training hyperparameters
    parser.add_argument('--imgsz', type=int, default=640, help="Image/video size for training (default: 640)")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument('--patience', type=int, default=5, help="Early stopping patience (default: 5)")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument('--optimizer', type=str, default="auto", help="optimizer (default=auto)")
    parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate, (default=1e-3)")
    parser.add_argument('--device', type=str, default="3", help="device to use for training (default 3)")
    parser.add_argument('--workers', type=int, default=8, help="Number of data loader workers (default: 8)")
    parser.add_argument('--verbose', type=int, default=0, help="Verbose training (default: 0)")
    parser.add_argument('--seed', type=int, default=0, help="Seed (default: 0)")

    # Resume training from a checkpoint
    parser.add_argument('--resume', type=str, default=None, help="Path to the model checkpoint to resume training from")

    return parser.parse_args()


def main():
    # Load environment variables from .env file
    load_dotenv()
    wandb_username = os.getenv("WANDB_USERNAME")
    wandb_api_key = os.getenv("WANDB_API_KEY")
    #
    # Ensure W&B API key is loaded
    if wandb_api_key is None:
        raise ValueError("W&B API key not found. Please check your .env file.")

    # Initialize W&B using the API key
    os.environ["WANDB_USERNAME"] = wandb_username
    os.environ["WANDB_API_KEY"] = wandb_api_key

    # Parse the command-line arguments
    args = parse_args()

    # Initialize the YOLO11 model
    if args.resume is not None:
        print(f"Resuming training from checkpoint: {args.resume}")
        resume = True
        model = YOLO(args.resume)  # Resume training from the checkpoint model
    else:
        print(f"Training from scratch or pretrained model: {args.model}")
        resume = False
        model = YOLO(args.model)  # Start training from scratch or a pretrained model

    wandb.login(key=wandb_api_key)

    # Train the model
    results = model.train(
        data=args.data,  # Path to the dataset YAML
        resume=resume,
        epochs=args.epochs,  # Number of epochs
        patience=args.patience,
        batch=args.batch_size,  # Batch size
        imgsz=args.imgsz,  # Image size for training
        save=True,  # Save best model
        save_period=1,  # Save model after each epoch
        device=args.device,
        workers=args.workers,  # Number of workers
        project="experiments",  # Project folder
        name=args.run_name,  # Run name
        exist_ok=True,  # overwrite run with same name
        optimizer=args.optimizer,
        verbose=bool(args.verbose),  # Verbose output
        seed=args.seed,
        single_cls=True,  # Single-class detection
        cos_lr=True,  # cosine annealing lr scheduler
        val=True,  # perform validation
        plots=True,  # create plots
    )

    # Finish the W&B run
    wandb.finish()


if __name__ == "__main__":
    main()

from ultralytics import YOLO

import wandb
from src.configs.track import check_tracking_args, preprocess_tracking_args
from src.configs.utils import parse_config_file, read_yaml_config
from src.logging.track import log_tracking_results
from src.logging.wandb import get_wandb_api_key, get_wandb_entity
from src.tracking.solutions import count_objects_in_video, heatmap_in_video


def main():
    # Parse the config file from command line
    config_file_path = parse_config_file()
    # Read YAML config file and transform it into a dict
    tracking_args = read_yaml_config(config_file_path)

    # Check arguments validity
    tracking_args = check_tracking_args(
        tracking_args
    )  # TODO argument checks - for correct or YOLO checks
    # Preprocess arguments based on input data format
    tracking_args = preprocess_tracking_args(tracking_args)  # TODO preprocessing

    print(tracking_args)
    count_objects_in_video(tracking_args.copy())
    heatmap_in_video(tracking_args.copy())


if __name__ == "__main__":
    main()

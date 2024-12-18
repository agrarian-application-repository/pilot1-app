from ultralytics import YOLO
import wandb

from src.configs.track import check_tracking_args
from src.configs.utils import parse_config_file, read_yaml_config


def main():
    # Parse the config file from command line
    config_file_path = parse_config_file()
    # Read YAML config file and transform it into a dict
    tracking_args = read_yaml_config(config_file_path)

    tracking_args = check_tracking_args(tracking_args)

    # Load the model
    model_checkpoint = tracking_args.pop("model")
    task = tracking_args.pop("task")
    model = YOLO(model=model_checkpoint, task=task)

    print("PERFORMING TRACKING WITH THE FOLLOWING ARGUMENTS:")
    print(tracking_args, "\n")

    # Perform tracking with the model
    results = model.track(**tracking_args)

    print("saving with stream")
    # iterate through generator to trigger saving of results
    for r in results:
        pass

    # reinsert popped arguments before logging
    tracking_args["model"] = model_checkpoint
    tracking_args["task"] = task

    # wandb_api_key = get_wandb_api_key()
    # wandb.login(key=wandb_api_key)

    # wandb_entity = get_wandb_entity()
    # wandb.init(entity=wandb_entity)

    wandb.init()
    wandb.config.update(tracking_args)
    wandb.finish()


if __name__ == "__main__":
    main()

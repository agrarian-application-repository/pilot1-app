from ultralytics import YOLO

import wandb
from src.configs.train import check_train_args
from src.configs.utils import parse_config_file, read_yaml_config
from src.logging.wandb_access import get_wandb_api_key, get_wandb_entity


def main():

    # Parse the config file from command line
    config_file_path = parse_config_file()
    # Read YAML config file and transform it into a dict
    train_args = read_yaml_config(config_file_path)

    train_args = check_train_args(train_args)

    print("PERFORMING TRAINING WITH THE FOLLOWING ARGUMENTS:")
    print(train_args, "\n")

    # Load the model
    model_checkpoint = train_args.pop("model")
    task = train_args.pop("task")
    model = YOLO(model=model_checkpoint, task=task)

    # wandb_api_key = get_wandb_api_key()
    # wandb.login(key=wandb_api_key)

    # wandb_entity = get_wandb_entity()
    # wandb.init(entity=wandb_entity)

    # Train model
    results = model.train(**train_args)

    # Finish the W&B run
    # wandb.finish()


if __name__ == "__main__":
    main()

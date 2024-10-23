from src.config_v2.hyperparameters_search import check_hs_args, preprocess_search_args
from src.config_v2.utils import parse_config_file, read_yaml_config
from ultralytics import YOLO

import wandb
from src.logging.wandb import get_wandb_api_key, get_wandb_entity


def main():
    # Parse the config file from command line
    config_file_path = parse_config_file()
    # Read YAML config file and transform it into a dict
    hs_args = read_yaml_config(config_file_path)

    # Check arguments validity
    hs_args = check_hs_args(
        hs_args
    )  # TODO argument checks - for correct or YOLO checks

    # Setup 'search' argument to contain tune.uniform() or tune.choice() ranges
    hs_args = preprocess_search_args(hs_args)

    print("PERFORMING HYPERPARAMETERS SEARCH WITH THE FOLLOWING ARGUMENTS:")
    print(hs_args, "\n")

    # Load the model
    model_checkpoint = hs_args.pop("model")
    model = YOLO(model_checkpoint)

    # wandb_api_key = get_wandb_api_key()
    # wandb.login(key=wandb_api_key)

    # wandb_entity = get_wandb_entity()
    # wandb.init(entity=wandb_entity)

    # Run hyperparameters search
    results_grid = model.tune(**hs_args)

    # Finish the W&B run
    wandb.finish()


if __name__ == "__main__":
    main()

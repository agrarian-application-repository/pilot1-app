# https://docs.ultralytics.com/integrations/ray-tune/

from src.cli.hyperparameters_search import manage_hyperparameters_search_arguments
from ultralytics import YOLO

import wandb
from src.logging.wandb import get_wandb_api_key, get_wandb_entity


def main():
    # Parse the command-line arguments
    search_args = manage_hyperparameters_search_arguments()

    model = YOLO(search_args.pop("model"))

    # wandb_api_key = get_wandb_api_key()
    # wandb.login(key=wandb_api_key)

    # wandb_entity = get_wandb_entity()
    # wandb.init(entity=wandb_entity)

    # Run hyperparameters search
    results_grid = model.tune(use_ray=True, **search_args)

    # Finish the W&B run
    wandb.finish()


if __name__ == "__main__":
    main()

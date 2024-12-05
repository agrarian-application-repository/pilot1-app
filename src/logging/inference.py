from typing import Any

import wandb
from src.logging.wandb_access import get_wandb_api_key, get_wandb_entity


# TODO implement
def log_inference_detection_results(results, args: dict[str: Any]) -> None:
    raise NotImplementedError("Segmentation evaluation logging not implemented yet")

    # wandb_api_key = get_wandb_api_key()
    # wandb.login(key=wandb_api_key)

    # wandb_entity = get_wandb_entity()
    # wandb.init(entity=wandb_entity)

    wandb.init()

    wandb.config.update(args)

    metrics = {
        # todo
    }

    # Finish the W&B run
    wandb.finish()


# TODO implement
def log_inference_segmentation_results(results, args: dict[str: Any]) -> None:
    raise NotImplementedError("Segmentation evaluation logging not implemented yet")

    # wandb_api_key = get_wandb_api_key()
    # wandb.login(key=wandb_api_key)

    # wandb_entity = get_wandb_entity()
    # wandb.init(entity=wandb_entity)

    wandb.init()

    wandb.config.update(args)

    metrics = {
        # todo
    }

    # Finish the W&B run
    wandb.finish()

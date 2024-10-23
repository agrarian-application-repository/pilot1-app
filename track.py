from ultralytics import YOLO

import wandb
from src.logging.wandb import get_wandb_api_key, get_wandb_entity
from src.logging.track import log_tracking_results

from src.configs.utils import parse_config_file, read_yaml_config
from src.configs.track import check_tracking_args, preprocess_tracking_args


def main():
    # Parse the config file from command line
    config_file_path = parse_config_file()
    # Read YAML config file and transform it into a dict
    tracking_args = read_yaml_config(config_file_path)

    # Check arguments validity
    tracking_args = check_tracking_args(tracking_args)  # TODO argument checks - for correct or YOLO checks
    # Preprocess arguments based on input data format
    tracking_args = preprocess_tracking_args(tracking_args)  # TODO preprocessing

    # Load the model
    model_checkpoint = tracking_args.pop("model")
    model = YOLO(model_checkpoint)

    print("PERFORMING TRACKING WITH THE FOLLOWING ARGUMENTS:")
    print(tracking_args, "\n")

    # Perform tracking with the model
    results = model.track(**tracking_args)

    if tracking_args["stream"]:
        print("saving with stream")
        # iterate through generator to trigger saving of results
        for r in results:
            pass

    # TODO log results
    # wandb_api_key = get_wandb_api_key()
    # wandb.login(key=wandb_api_key)

    # wandb_entity = get_wandb_entity()
    # wandb.init(entity=wandb_entity)

    # tracking_args["model"] = model_checkpoint # re-insert before logging
    # wandb.config.update(**inference_args)

    # log_tracking_results(results)

    # Finish the W&B run
    # wandb.finish()


if __name__ == "__main__":
    main()

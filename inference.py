from ultralytics import YOLO

import wandb
from src.configs.inference import (check_inference_args,
                                   preprocess_inference_args)
from src.configs.utils import parse_config_file, read_yaml_config
from src.logging.inference import log_inference_results
from src.logging.wandb import get_wandb_api_key, get_wandb_entity


def main():
    # Parse the config file from command line
    config_file_path = parse_config_file()
    # Read YAML config file and transform it into a dict
    inference_args = read_yaml_config(config_file_path)

    # Check arguments validity
    inference_args = check_inference_args(
        inference_args
    )  # TODO argument checks - for correct or YOLO checks
    # Preprocess arguments based on input data format
    inference_args = preprocess_inference_args(inference_args)  # TODO preprocessing

    print("PERFORMING INFERENCE WITH THE FOLLOWING ARGUMENTS:")
    print(inference_args, "\n")

    # Load the model
    model_checkpoint = inference_args.pop("model")
    model = YOLO(model_checkpoint)

    # Perform inference with the model
    results = model.predict(**inference_args)

    if inference_args["stream"]:
        print("saving with stream")
        # iterate through generator to trigger saving of results
        for r in results:
            pass

    # TODO(?) logging predictions on wandb
    """
    # wandb_api_key = get_wandb_api_key()
    # wandb.login(key=wandb_api_key)

    # wandb_entity = get_wandb_entity()
    # wandb.init(entity=wandb_entity)
    wandb.init()

    inference_args["model"] = model_checkpoint  # re-insert before logging
    wandb.config.update(inference_args)

    # Finish the W&B run
    wandb.finish()
    """


if __name__ == "__main__":
    main()

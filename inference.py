from ultralytics import YOLO
import wandb

from src.configs.inference import check_inference_args
from src.configs.utils import parse_config_file, read_yaml_config


def main():

    # Parse the config file from command line
    config_file_path = parse_config_file()
    # Read YAML config file and transform it into a dict
    inference_args = read_yaml_config(config_file_path)

    inference_args = check_inference_args(inference_args)

    print("PERFORMING INFERENCE WITH THE FOLLOWING ARGUMENTS:")
    print(inference_args, "\n")

    # Load the model
    model_checkpoint = inference_args.pop("model")
    task = inference_args.pop("task")
    model = YOLO(model=model_checkpoint, task=task)

    # Perform inference with the model
    results = model.predict(**inference_args)

    if inference_args["stream"]:
        print("saving with stream")
        # iterate through generator to trigger saving of results
        for r in results:
            pass

    # reinsert popped arguments before logging
    inference_args["model"] = model_checkpoint
    inference_args["task"] = task

    # wandb_api_key = get_wandb_api_key()
    # wandb.login(key=wandb_api_key)

    # wandb_entity = get_wandb_entity()
    # wandb.init(entity=wandb_entity)

    wandb.init()
    wandb.config.update(inference_args)
    wandb.finish()


if __name__ == "__main__":
    main()

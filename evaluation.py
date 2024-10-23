from src.config_v2.evaluate import check_eval_args
from src.config_v2.utils import parse_config_file, read_yaml_config
from ultralytics import YOLO

import wandb
from src.logging.evaluation import log_eval_metrics
from src.logging.wandb import get_wandb_api_key, get_wandb_entity


def main():
    # Parse the config file from command line
    config_file_path = parse_config_file()
    # Read YAML config file and transform it into a dict
    eval_args = read_yaml_config(config_file_path)

    # Check arguments validity
    eval_args = check_eval_args(
        eval_args
    )  # TODO argument checks - for correct or YOLO checks

    print("PERFORMING EVALUATION WITH THE FOLLOWING ARGUMENTS:")
    print(eval_args, "\n")

    # Load the model
    model_checkpoint = eval_args.pop("model")
    model = YOLO(model_checkpoint)

    # Evaluate the model
    results = model.val(**eval_args)

    # TODO log results
    # wandb_api_key = get_wandb_api_key()
    # wandb.login(key=wandb_api_key)

    # wandb_entity = get_wandb_entity()
    # wandb.init(entity=wandb_entity)

    # eval_args["model"] = model_checkpoint # re-insert before logging
    # wandb.config.update(**eval_args)

    # log_eval_metrics(results)

    # Finish the W&B run
    # wandb.finish()


if __name__ == "__main__":
    main()

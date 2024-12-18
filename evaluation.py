from ultralytics import YOLO

from src.configs.evaluate import check_eval_args
from src.configs.utils import parse_config_file, read_yaml_config
from src.logging.evaluation import log_eval_detection_results, log_eval_segmentation_results


def main() -> None:

    # Parse the config file from command line
    config_file_path = parse_config_file()
    # Read YAML config file and transform it into a dict
    eval_args = read_yaml_config(config_file_path)

    eval_args = check_eval_args(eval_args)

    print("PERFORMING EVALUATION WITH THE FOLLOWING ARGUMENTS:")
    print(eval_args, "\n")

    # Load the model
    model_checkpoint = eval_args.pop("model")
    task = eval_args.pop("task")
    model = YOLO(model=model_checkpoint, task=task)

    # Evaluate the model
    results = model.val(**eval_args)

    # reinsert popped arguments
    eval_args["model"] = model_checkpoint  # re-insert before logging
    eval_args["task"] = task  # re-insert before logging

    # only base results:
    if task == "detect":
        log_eval_detection_results(results, eval_args)
    elif task == "segment":
        log_eval_segmentation_results(results, eval_args)


if __name__ == "__main__":
    main()

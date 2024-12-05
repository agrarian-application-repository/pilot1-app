from ultralytics import YOLO

from src.configs.track import check_tracking_args, preprocess_tracking_args
from src.configs.utils import parse_config_file, read_yaml_config
from src.logging.track import (log_tracking_detection_results,
                               log_tracking_segmentation_results)


def main():
    # Parse the config file from command line
    config_file_path = parse_config_file()
    # Read YAML config file and transform it into a dict
    tracking_args = read_yaml_config(config_file_path)

    # TODO Check arguments validity
    tracking_args = check_tracking_args(tracking_args)
    # TODO Preprocess arguments based on input data format
    tracking_args = preprocess_tracking_args(tracking_args)

    # Load the model
    model_checkpoint = tracking_args.pop("model")
    task = tracking_args.pop("task")
    model = YOLO(model=model_checkpoint, task=task)

    print("PERFORMING TRACKING WITH THE FOLLOWING ARGUMENTS:")
    print(tracking_args, "\n")

    # Perform tracking with the model
    results = model.track(**tracking_args)

    if tracking_args["stream"]:
        print("saving with stream")
        # iterate through generator to trigger saving of results
        for r in results:
            pass

    # reinsert popped arguments before logging
    tracking_args["model"] = model_checkpoint
    tracking_args["task"] = task

    # TODO logging
    # if task == "detect":
    #     log_tracking_detection_results(results, inference_args)
    # elif task == "segment":
    #    log_tracking_segmentation_results(results, inference_args)


if __name__ == "__main__":
    main()

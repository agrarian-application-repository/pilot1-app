from ultralytics import YOLO

from src.configs.inference import check_inference_args, preprocess_inference_args
from src.configs.utils import parse_detect_segment_config_files, read_yaml_config
from src.in_danger import perform_in_danger_analysis


def main():
    # Parse the config files from command line
    detect_config_file_path, segment_config_file_path = (
        parse_detect_segment_config_files()
    )

    # Read YAML config files and transform them into dicts
    detection_args = read_yaml_config(detect_config_file_path)
    segmentation_args = read_yaml_config(segment_config_file_path)

    # Check arguments validity
    # detection_args = check_detection_args(detection_args)  TODO checks
    # segmentation_args = check_segmentation_args(segmentation_args)   TODO checks

    # Preprocess arguments based on input data format
    # detection_args = preprocess_detection_args(detection_args)  # TODO preprocessing
    # segmentation_args = preprocess_segmentation_args(segmentation_args)  # TODO preprocessing

    print("PERFORMING IN-DANGER WITH THE FOLLOWING ARGUMENTS:")
    print("SHEEP DETECTION:")
    print(detection_args)
    print("DANGEROUS TERRAIN SEGMENTATION:")
    print(segmentation_args)
    print("\n")

    # Load the detection model
    detection_model_checkpoint = detection_args.pop("model")
    detection_model = YOLO(detection_model_checkpoint)

    # Load the segmentation model
    segmentation_model_checkpoint = segmentation_args.pop("model")
    segmentation_model = YOLO(segmentation_model_checkpoint)

    perform_in_danger_analysis()


if __name__ == "__main__":
    main()

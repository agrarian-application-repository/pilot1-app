from src.configs.in_danger import check_in_danger_args, preprocess_in_danger_args
from src.configs.utils import parse_config_file, read_yaml_config
from src.in_danger.in_danger import perform_in_danger_analysis


def main():
    # Parse the config files from command line
    in_danger_config_file = parse_config_file()

    # Read YAML config files and transform them into dicts
    in_danger_args = read_yaml_config(in_danger_config_file)

    # TODO Check arguments validity
    # in_danger_args = check_in_danger_args(in_danger_args)

    # TODO Preprocess arguments based on input data format
    # in_danger_args = preprocess_in_danger_args(in_danger_args)
    # TODO ASSERT sensor_width_mm/sensor_height_mm == sensor_width_pixels/sensor_height_pixels

    print("PERFORMING IN-DANGER WITH THE FOLLOWING ARGUMENTS:")
    print(in_danger_args)
    print("\n")

    perform_in_danger_analysis(
        input_args=in_danger_args["input"],
        output_args=in_danger_args["output"],
        detection_args=in_danger_args["detection"],
        segmentation_args=in_danger_args["segmentation"],
    )


if __name__ == "__main__":
    main()

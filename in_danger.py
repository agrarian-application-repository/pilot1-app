from src.configs.in_danger import check_in_danger_args, preprocess_in_danger_args
from src.configs.utils import parse_config_file, read_yaml_config
from src.in_danger.in_danger import perform_in_danger_analysis
import cProfile
import pstats


def main():

    # Parse the config files from command line
    input_config_file = parse_config_file()

    # Read YAML config files and transform them into dicts
    input_args = read_yaml_config(input_config_file)

    # TODO Check arguments validity
    # input_args = check_in_danger_input_args(input_args)
    # TODO Preprocess arguments based on input data format
    # input_args = preprocess_in_danger_input_args(input_args)

    drone_args = read_yaml_config("configs/drone_specs.yaml")
    # TODO Check arguments validity
    # drone_args = check_drone_args(drone_args)
    # TODO Preprocess arguments based on input data format
    # drone_args = preprocess_drone_args(drone_args)
    # TODO ASSERT sensor_width_mm/sensor_height_mm == sensor_width_pixels/sensor_height_pixels

    output_args = read_yaml_config("configs/in_danger/output.yaml")
    detection_args = read_yaml_config("configs/in_danger/detector.yaml")
    segmentation_args = read_yaml_config("configs/in_danger/segmenter.yaml")

    print("PERFORMING IN-DANGER WITH THE FOLLOWING ARGUMENTS:")
    print("Input arguments")
    print(input_args)
    print("Output arguments")
    print(output_args)
    print("Detector arguments")
    print(detection_args)
    print("Segmenter arguments")
    print(segmentation_args)
    print("Drone arguments")
    print(drone_args)
    print("\n")

    perform_in_danger_analysis(
        input_args=input_args,
        output_args=output_args,
        detection_args=detection_args,
        segmentation_args=segmentation_args,
        drone_args=drone_args,
    )


if __name__ == "__main__":

    # cProfile.run('main()', './profile_output.prof')
    main()

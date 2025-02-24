# from src.configs.health_monitoring import check_health_monitoring_args, preprocess_health_monitoring_args
from src.configs.utils import parse_config_file, read_yaml_config
from src.health_monitoring.health_monitoring import perform_health_monitoring_analysis


def main():
    # Parse the config files from command line
    input_config_file = parse_config_file()

    # Read YAML config files and transform them into dicts
    input_args = read_yaml_config(input_config_file)

    # TODO Check arguments validity
    # input_args = check_health_monitoring_input_args(input_args)
    # TODO Preprocess arguments based on input data format
    # input_args = preprocess_health_monitoring_input_args(input_args)

    drone_args = read_yaml_config("configs/drone_specs.yaml")
    # TODO Check arguments validity
    # drone_args = check_drone_args(drone_args)
    # TODO Preprocess arguments based on input data format
    # drone_args = preprocess_drone_args(drone_args)
    # TODO ASSERT sensor_width_mm/sensor_height_mm == sensor_width_pixels/sensor_height_pixels

    output_args = read_yaml_config("configs/health_monitoring/output.yaml")
    tracking_args = read_yaml_config("configs/health_monitoring/tracker.yaml")
    anomaly_detection_args = read_yaml_config("configs/health_monitoring/anomaly_detector.yaml")

    print("PERFORMING HEALTH MONITORING WITH THE FOLLOWING ARGUMENTS:")
    print("Input arguments")
    print(input_args)
    print("Output arguments")
    print(output_args)
    print("Tracker arguments")
    print(tracking_args)
    print("Anomaly detection arguments")
    print(anomaly_detection_args)
    print("Drone arguments")
    print(drone_args)
    print("\n")

    perform_health_monitoring_analysis(
        input_args=input_args,
        output_args=output_args,
        tracking_args=tracking_args,
        anomaly_detection_args=anomaly_detection_args,
        drone_args=drone_args,
    )


if __name__ == "__main__":
    main()

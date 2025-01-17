# from src.configs.health_monitoring import check_health_monitoring_args, preprocess_health_monitoring_args
from src.configs.utils import parse_config_file, read_yaml_config
from src.health_monitoring import perform_health_monitoring_analysis


def main():
    # Parse the config files from command line
    health_monitoring_config_file = parse_config_file()

    # Read YAML config files and transform them into dicts
    health_monitoring_args = read_yaml_config(health_monitoring_config_file)

    # TODO Check arguments validity
    # health_monitoring_args = check_health_monitoring_args(health_monitoring_args)

    # TODO Preprocess arguments based on input data format
    # health_monitoring_args = preprocess_health_monitoring_args(health_monitoring_args)

    print("PERFORMING HEALTH MONITORING WITH THE FOLLOWING ARGUMENTS:")
    print(health_monitoring_args)
    print("\n")

    perform_health_monitoring_analysis(
        input_args=health_monitoring_args["input"],
        output_args=health_monitoring_args["output"],
        detection_args=health_monitoring_args["tracking"],
        anomaly_args=health_monitoring_args["anomaly_detection"],
    )


if __name__ == "__main__":
    main()

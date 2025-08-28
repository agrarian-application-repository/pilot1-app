from src.configs.health_monitoring import check_health_monitoring_args
from src.configs.drone import check_drone_args
from src.configs.networks import check_networking_args
from src.configs.utils import read_yaml_config
from src.health_monitoring.health_monitoring_stream import perform_health_monitoring
import os

def main():

    # Read input YAML config file and transform it into dict
    input_args = read_yaml_config("configs/health_monitoring/input.yaml")
    # Check validity of arguments
    input_args = check_health_monitoring_args(input_args)   # TODO: implement

    # Read drone YAML config file and transform it into dict
    drone_args = read_yaml_config("configs/drone_specs.yaml")
    # Check validity of arguments
    drone_args = check_drone_args(drone_args)

    output_args = read_yaml_config("configs/health_monitoring/output.yaml")
    tracking_args = read_yaml_config("configs/health_monitoring/tracker.yaml")
    anomaly_detection_args = read_yaml_config("configs/health_monitoring/anomaly_detector.yaml")

    # URLs cvomponents are passed to the container as environmental variables
    urls = {
        "stream_ip": os.environ.get("STREAM_IP", "mediamtx_server"),  
        "stream_port": int(os.environ.get("STREAM_PORT", "1935")),
        "stream_name": os.environ.get("STREAM_NAME", "drone"),
        "telemetry_ip": os.environ.get("TELEMETRY_IP", "0.0.0.0"),
        "telemetry_port": int(os.environ.get("TELEMETRY_PORT", "12345")),
        "annotations_ip": os.environ.get("ANNOTATIONS_IP", "mediamtx_server"),
        "annotations_port": int(os.environ.get("ANNOTATIONS_PORT", "8554")),
        "annotations_name": os.environ.get("ANNOTATIONS_NAME", "annot"),
        "alerts_ip": os.environ.get("ALERTS_IP", "127.0.0.1"),
        "alerts_port": int(os.environ.get("ALERTS_PORT", "54321")),
    }
    check_networking_args(urls)

    input_args["source"] = f"rtmp://{urls["stream_ip"]}:{urls["stream_port"]}/{urls["stream_name"]}"
    input_args["telemetry_in_port"] = urls["telemetry_port"]
    
    output_args["video_url_out"] = f"rtsp://{urls["annotations_ip"]}:{urls["annotations_port"]}/{urls["annotations_name"]}"
    output_args["alerts_url_out"] = f"tcp://{urls["alerts_ip"]}:{urls["alerts_port"]}"

    # output is saved to the /app/outputs folder, exported for visibility as volume
    output_args["output_dir"] = "outputs/" + output_args["output_dir"]

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

    perform_health_monitoring(
        input_args=input_args,
        output_args=output_args,
        tracking_args=tracking_args,
        anomaly_detection_args=anomaly_detection_args,
        drone_args=drone_args,
    )


if __name__ == "__main__":
    main()

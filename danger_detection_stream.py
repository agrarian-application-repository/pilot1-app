from src.configs.danger_detection import check_danger_detection_args
from src.configs.drone import check_drone_args
from src.configs.utils import read_yaml_config
from src.danger_detection.danger_detection_stream import perform_danger_detection
import os

def main():

    # Read input YAML config file and transform it into dict
    input_args = read_yaml_config("configs/danger_detection/input.yaml")
    # Check validity of arguments
    input_args = check_danger_detection_args(input_args)

    # Read drone YAML config file and transform it into dict
    drone_args = read_yaml_config("configs/drone_specs.yaml")
    # Check validity of arguments
    drone_args = check_drone_args(drone_args)

    output_args = read_yaml_config("configs/danger_detection/output.yaml")
    detection_args = read_yaml_config("configs/danger_detection/detector.yaml")
    segmentation_args = read_yaml_config("configs/danger_detection/segmenter.yaml")

    output_args["output_dir"] = "outputs/" + output_args["output_dir"]

    urls = {
        "stream_ip": os.environ.get("stream_ip", "127.0.0.1"),  
        "stream_port": int(os.environ.get("stream_port", "1935")),
        "stream_name": os.environ.get("stream_name", "drone"),
        "telemetry_ip": os.environ.get("telemetry_ip", "0.0.0.0"),
        "telemetry_port": int(os.environ.get("telemetry_port", "12345")),
        "annotations_ip": os.environ.get("annotations_ip", "127.0.0.1"),
        "annotations_port": int(os.environ.get("annotations_port", "8554")),
        "annotations_name": os.environ.get("annotations_name", "annot"),
        "alerts_ip": os.environ.get("alerts_ip", "127.0.0.1"),
        "alerts_port": int(os.environ.get("alerts_port", "54321")),
    }
    # TODO
    # check_urls

    input_args["source"] = f"rtmp://{urls["stream_ip"]}:{urls["stream_port"]}/{urls["stream_name"]}"
    input_args["telemetry_in_port"] = urls["telemetry_port"]
    
    output_args["video_url_out"] = f"rtmp://{urls["annotations_ip"]}:{urls["annotations_port"]}/{urls["annotations_name"]}"
    output_args["alerts_url_out"] = f"tcp://{urls["alerts_ip"]}:{urls["alerts_port"]}"
   

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

    perform_danger_detection(
        input_args=input_args,
        output_args=output_args,
        detection_args=detection_args,
        segmentation_args=segmentation_args,
        drone_args=drone_args,
    )


if __name__ == "__main__":

    main()

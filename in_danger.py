from src.configs.in_danger import check_in_danger_args
from src.configs.drone import check_drone_args
from src.configs.utils import read_yaml_config
from src.in_danger.in_danger_parallel import perform_in_danger_analysis


def main():

    # Read input YAML config file and transform it into dict
    input_args = read_yaml_config("configs/in_danger/input.yaml")
    # Check validity of arguments
    input_args = check_in_danger_args(input_args)

    # TODO this will be passed through the container either as env variable or volumes (to remove later)
    # -------------------------------------------------
    # str: Data source (a video) for in-danger analysis.
    input_args["source"] = '/archive/group/ai/datasets/AGRARIAN/MAICH_v1/DJI_20241024104935_0008_D.MP4'
    # str: Drone metadata file (.srt)
    input_args["flight_data"] = '/archive/group/ai/datasets/AGRARIAN/MAICH_v1/DJI_20241024104935_0008_D.SRT'
    # str: Path to the DEM data
    # dem: '/archive/group/ai/datasets/AGRARIAN/DEM/merged/merged.tif'
    input_args["dem"] = '/archive/group/ai/datasets/AGRARIAN/DEM/Copernicus_DSM_04_N35_00_E024_00_DEM.tif'
    # str or null: Path to the DEM data mask, if null assumes all pixels are valid
    # dem_mask: '/archive/group/ai/datasets/AGRARIAN/DEM/merged/merged_mask.tif'
    input_args["dem_mask"] = None
    # int: Frame stride for video inputs.
    # Allows skipping frames to speed up inference. Higher values skip more frames.
    # Range: Any positive integer.
    input_args["vid_stride"] = 1
    # -------------------------------------------------

    # Read drone YAML config file and transform it into dict
    drone_args = read_yaml_config("configs/drone_specs.yaml")
    # Check validity of arguments
    drone_args = check_drone_args(drone_args)

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

    main()

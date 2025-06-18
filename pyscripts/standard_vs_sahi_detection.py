from ultralytics import YOLO
from src.configs.utils import parse_config_file, read_yaml_config
from src.in_danger.detection.detection import setup_detecion_model_sahi, perform_detection_sahi, perform_detection
import cv2


def main():

    # Read YAML config file and transform it into a dict

    inference_args = read_yaml_config("configs/in_danger/detector.yaml")
    print(inference_args)

    task = "detect"
    print("%%%%%%%%%%%%%%%%%%%")

    # 00d41c29-DJI_20250306132455_0001_D_9492.png
    # 0f1f2b17-DJI_20250224124109_0004_D_9040.png
    # 6aee49f6-DJI_20250318124342_0001_D_0.png
    frame = cv2.imread("/archive/group/ai/datasets/AGRARIAN/manual_annotations/test_splitting/images/6aee49f6-DJI_20250318124342_0001_D_0.png")

    ############## standard #################
    detection_args = inference_args.copy()
    model_checkpoint = detection_args.pop("model_checkpoint")
    detector = YOLO(model=model_checkpoint, task=task)
    detector.to(detection_args["device"])
    classes, boxes_centers, boxes_corner1, boxes_corner2 = perform_detection(detector, frame, detection_args)
    print("classes")
    print(classes)
    print("boxes_centers")
    print(boxes_centers)
    print("boxes_corner1")
    print(boxes_corner1)
    print("boxes_corner2")
    print(boxes_corner2)

    ############# SAHI ###################
    sahi_detection_args = inference_args.copy()
    sahi_detector = setup_detecion_model_sahi(sahi_detection_args)
    classes, boxes_centers, boxes_corner1, boxes_corner2 = perform_detection_sahi(sahi_detector, frame, sahi_detection_args)
    print(len(classes))
    print("classes")
    print(classes)
    print("boxes_centers")
    print(boxes_centers)
    print("boxes_corner1")
    print(boxes_corner1)
    print("boxes_corner2")
    print(boxes_corner2)


if __name__ == "__main__":
    main()

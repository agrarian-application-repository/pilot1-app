from ultralytics import YOLO
from src.configs.utils import read_yaml_config
from src.danger_detection.segmentation.segmentation import perform_segmentation, setup_segmentation_model_sahi, perform_segmentation_sahi
import cv2
import numpy as np


def main():

    # Read YAML config file and transform it into a dict

    inference_args = read_yaml_config("configs/danger_detection/segmenter.yaml")
    print(inference_args)

    task = "segment"
    print("%%%%%%%%%%%%%%%%%%%")

    # 92ea8b51-DJI_20241024103403_0001_D_0.png
    # 8271871a-DJI_20241024103403_0001_D_1808.png
    # 6aee49f6-DJI_20250318124342_0001_D_0.png
    frame = cv2.imread("/archive/group/ai/datasets/AGRARIAN/manual_annotations/test_splitting/images/6aee49f6-DJI_20250318124342_0001_D_0.png")

    ############## standard #################
    segmentation_args = inference_args.copy()
    model_checkpoint = segmentation_args.pop("model_checkpoint")
    segmenter = YOLO(model=model_checkpoint, task=task)
    segmenter.to(segmentation_args["device"])
    mask = perform_segmentation(segmenter, frame, segmentation_args)
    print(mask.shape)
    print(mask.dtype)
    print(np.max(mask))
    cv2.imwrite("TMP/standard2.png", mask*255)

    ############# SAHI ###################
    sahi_segmentation_args = inference_args.copy()
    sahi_segmenter = setup_segmentation_model_sahi(sahi_segmentation_args)
    mask = perform_segmentation_sahi(sahi_segmenter, frame, sahi_segmentation_args)
    print(mask.shape)
    print(mask.dtype)
    print(np.max(mask))
    cv2.imwrite("TMP/sahi2.png", mask*255)

if __name__ == "__main__":
    main()

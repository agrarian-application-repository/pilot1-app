import numpy as np
from sahi.predict import get_sliced_prediction
from src.in_danger.sahi_yolo import ModifiedUltralyticsDetectionModel


def perform_detection(detector, frame, detection_args):
    # Detect animals in frame
    detection_results = detector.predict(source=frame, **detection_args)
    return postprocess_detection_results(detection_results)


def postprocess_detection_results(detection_results):

    if detection_results[0].boxes is not None:
        # Parse detection results to get bounding boxes
        classes = detection_results[0].boxes.cls.cpu().numpy().astype(int)
        xywh_boxes = detection_results[0].boxes.xywh.cpu().numpy().astype(int)
        xyxy_boxes = detection_results[0].boxes.xyxy.cpu().numpy().astype(int)

        # Create additional variables to store useful info from the detections
        boxes_centers = xywh_boxes[:, :2]
        boxes_corner1 = xyxy_boxes[:, :2]
        boxes_corner2 = xyxy_boxes[:, 2:]
    else:
        classes = np.array([], dtype=int)
        boxes_centers = np.array([], dtype=int)
        boxes_corner1 = np.array([], dtype=int)
        boxes_corner2 = np.array([], dtype=int)

    return classes, boxes_centers, boxes_corner1, boxes_corner2


def setup_detecion_model_sahi(detection_args):
    detection_model_checkpoint = detection_args.pop("model_checkpoint")
    detection_model = ModifiedUltralyticsDetectionModel(
        model_path=detection_model_checkpoint,
        task="detect",
        prediction_args=detection_args,
    )
    return detection_model


def perform_detection_sahi(detector, frame, detection_args):

    sahi_result = get_sliced_prediction(
        image=frame,
        detection_model=detector,
        slice_height=detection_args["imgsz"][0],
        slice_width=detection_args["imgsz"][1],
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        perform_standard_pred=False,
    )
    classes, boxes_centers, boxes_corner1, boxes_corner2 = postprocess_detection_results_sahi(sahi_result)
    return classes, boxes_centers, boxes_corner1, boxes_corner2


def postprocess_detection_results_sahi(sahi_result):

    # print("PREDICTIONS")
    object_predictions = sahi_result.object_prediction_list
    # print(object_predictions)

    # print("PROCESSED PREDICTIONS")
    if len(object_predictions) > 0:

        xyxy_boxes = np.array([obj_ann.bbox.to_xyxy() for obj_ann in object_predictions])

        classes = np.array([obj_ann.category.id for obj_ann in object_predictions])
        boxes_corner1 = xyxy_boxes[:, :2]
        boxes_corner2 = xyxy_boxes[:, 2:]
        boxes_centers = (boxes_corner1 + boxes_corner2) / 2

        boxes_corner1 = boxes_corner1.astype(int)
        boxes_corner2 = boxes_corner2.astype(int)
        boxes_centers = boxes_centers.astype(int)

    else:
        classes = np.array([], dtype=int)
        boxes_centers = np.array([], dtype=int)
        boxes_corner1 = np.array([], dtype=int)
        boxes_corner2 = np.array([], dtype=int)

    return classes, boxes_centers, boxes_corner1, boxes_corner2

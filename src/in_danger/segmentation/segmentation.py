import numpy as np
import cv2
from sahi.predict import get_sliced_prediction
from src.in_danger.sahi_yolo import ModifiedUltralyticsDetectionModel


def perform_segmentation(segmenter, frame, segmentation_args):
    # Highlight dangerous objects
    segment_results = segmenter.predict(source=frame, **segmentation_args)
    # frame size (H, W, 3)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    return postprocess_segmentation_results(segment_results, frame_height, frame_width)


def postprocess_segmentation_results(segment_results, frame_height, frame_width):

    if segment_results[0].masks is not None:  # danger found in the frame
        masks = segment_results[0].masks.data.int().cpu().numpy()
        segment_danger_mask = np.any(masks, axis=0).astype(np.uint8)    # merge the masks into one
        segment_danger_mask = cv2.resize(segment_danger_mask, dsize=(frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

    else:  # mask not found in frame
        segment_danger_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

    return segment_danger_mask


def setup_segmentation_model_sahi(segmentation_args):
    segmentation_model_checkpoint = segmentation_args.pop("model_checkpoint")
    segmentation_model = ModifiedUltralyticsDetectionModel(
        model_path=segmentation_model_checkpoint,
        task="segment",
        prediction_args=segmentation_args,
    )
    return segmentation_model


def perform_segmentation_sahi(segmenter, frame, segmentation_args):

    sahi_result = get_sliced_prediction(
        image=frame,
        detection_model=segmenter,
        slice_height=segmentation_args["imgsz"][0],
        slice_width=segmentation_args["imgsz"][1],
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        perform_standard_pred=False,
    )

    # frame size (H, W, 3)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    segment_danger_mask = postprocess_segmentation_results_sahi(sahi_result,frame_height, frame_width)
    return segment_danger_mask


def postprocess_segmentation_results_sahi(sahi_result, frame_height, frame_width):

    segment_danger_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

    object_predictions = sahi_result.object_prediction_list
    if len(object_predictions) > 0:
        masks = [obj_ann.mask.get_shifted_mask().bool_mask.astype(np.uint8) for obj_ann in object_predictions if obj_ann.mask is not None]
        for mask in masks:
            segment_danger_mask |= mask

    return segment_danger_mask

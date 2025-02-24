import numpy as np
import cv2


def perform_segmentation(segmenter, frame, segmentation_args):

    # Highlight dangerous objects
    segment_results = segmenter.predict(source=frame, **segmentation_args)

    # frame size (H, W, 3)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    if segment_results[0].masks is not None:  # danger found in the frame
        masks = segment_results[0].masks.data.int().cpu().numpy()
        segment_danger_mask = np.any(masks, axis=0).astype(np.uint8)    # merge the masks into one
        segment_danger_mask = cv2.resize(segment_danger_mask, dsize=(frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

    else:  # mask not found in frame
        segment_danger_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

    return segment_danger_mask

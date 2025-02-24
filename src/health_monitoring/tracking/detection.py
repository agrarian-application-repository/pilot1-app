def perform_tracking(detector, frame, tracking_args, aspect_ratio):
    # track animals in frame
    tracking_results = detector.track(source=frame, stream=True, persist=True, **tracking_args)

    # Parse detection results to get bounding boxes and
    # create additional variables to store useful info

    classes = tracking_results[0].boxes.cls.cpu().numpy().astype(int)

    xywh_boxes = tracking_results[0].boxes.xywh.cpu().numpy()
    xyxy_boxes = tracking_results[0].boxes.xyxy.cpu().numpy()
    xywhn_boxes = tracking_results[0].boxes.xywhn.cpu().numpy()

    boxes_corner1 = xyxy_boxes[:, :2].astype(int)
    boxes_corner2 = xyxy_boxes[:, 2:].astype(int)

    boxes_centers = xywh_boxes[:, :2].astype(int)
    normalized_boxes_centers = xywhn_boxes[:, :2].astype(int)

    scaled_normalized_boxes_centers = xywhn_boxes[:, :2]
    scaled_normalized_boxes_centers[:, 1] = scaled_normalized_boxes_centers[:, 1] / aspect_ratio
    scaled_normalized_boxes_centers = scaled_normalized_boxes_centers.astype(int)
    # both x and y normalized over their respctive lenght, but dimensions are different
    # when animal moves by 1080 pixels in diagonal ...
    # its normalized position would be (1080/1920, 1.0) ...
    # but i want it to be (1080/1920, 1080/1920) ...
    # to preserve distances given pixels on X and Y have the same size

    # Parse tracking ID
    if tracking_results[0].boxes.id is not None:
        ids_list = tracking_results[0].boxes.id.int().cpu().tolist()
    else:   # TODO is no tracking = no detections? (no)
        ids_list = []

    return_args = (
        ids_list,
        classes,
        boxes_centers,
        normalized_boxes_centers,
        scaled_normalized_boxes_centers,
        boxes_corner1,
        boxes_corner2,
    )

    return return_args

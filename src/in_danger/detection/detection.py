def perform_detection(detector, frame, detection_args):
    # Detect animals in frame
    detection_results = detector.predict(source=frame, **detection_args)

    # Parse detection results to get bounding boxes
    classes = detection_results[0].boxes.cls.cpu().numpy().astype(int)
    xywh_boxes = detection_results[0].boxes.xywh.cpu().numpy().astype(int)
    xyxy_boxes = detection_results[0].boxes.xyxy.cpu().numpy().astype(int)

    # Create additional variables to store useful info from the detections
    boxes_centers = xywh_boxes[:, :2]
    boxes_corner1 = xyxy_boxes[:, :2]
    boxes_corner2 = xyxy_boxes[:, 2:]

    return classes, boxes_centers, boxes_corner1, boxes_corner2

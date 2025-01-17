import cv2

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
PURPLE = (128, 0, 128)
CLASS_COLOR = [BLUE, PURPLE]


class HistoryTracker:
    """
     the object should act as follows. when  initialized, the input argument should be a lenght value.
     each key should comprise of two arrays, a list of value couples (x,y), an a boolean array indicating wheter the tuple at that position is valid.
     The calss should also have an update function that, given a list of ids and the corresponding coordinates, inserts the tuple in the array and sets it as valid.
     all other keys that are stored by the object but do not appear in the provided new set of id/tuples pairs, should replicate the last value and set the boolean mask value as False to indicate the data is artificial.
     If a key that never appeared before appears in the update, that key is added and a list/mask of lenght equal to the initially specified lenght argument is created for that key, with all values in the mask excet the last one being false, and the list of couples is comprised of the same tuple repeated LEN times.
     Finally, at each update step the oldest value in the list of tuples/masks should be dropped similarly to a fifo queue where LEN acts as the queu lenght)
    """

    def __init__(self, window_size):
        # window_size indicates the size of the value and mask arrays
        self.window_size = window_size
        self.data = {}

    def update(self, ids, coordinates):

        """
        Update the dictionary with new tuples and boolean masks.
        - ids: List of keys to update
        - coordinates: List of (x, y) tuples corresponding to the ids
        """

        updated_keys = []

        # Iterate over the ids and coordinates provided in the update
        for i, (key, coords) in enumerate(zip(ids, coordinates)):

            if key not in self.data:
                # If the key is new, initialize its tuple list and mask array
                self.data[key] = {}
                self.data[key]["coords"] = [coords] * self.window_size
                self.data[key]["valid"] = [False] * (self.window_size - 1) + [True]
            else:
                # If the key already exists:
                # Add the new value and set the validity indicator to True
                self.data[key]["coords"].append(coords)
                self.data[key]["valid"].append(True)
                # And remove the oldest value to maintain the window size
                self.data[key]["coords"].pop(0)
                self.data[key]["valid"].pop(0)

            updated_keys.append(key)

        non_updated_keys = set(self.data.keys()) - set(updated_keys)

        for key in non_updated_keys:
            # Repeat the last value, setting validity indicator to False
            last_value = self.data[key]["coords"][-1]
            self.data[key]["coords"].append(last_value)
            self.data[key]["valid"].append(False)
            # And remove the oldest value to maintain the window size
            self.data[key]["coords"].pop(0)
            self.data[key]["valid"].pop(0)


def perform_tracking(detector, history: HistoryTracker, frame, tracking_args):
    # track animals in frame
    tracking_results = detector.track(source=frame, stream=True, persist=True, **tracking_args)

    # Parse detection results to get bounding boxes
    classes = tracking_results[0].boxes.cls.cpu().numpy()
    xywh_boxes = tracking_results[0].boxes.xywh.cpu().numpy().astype(int)
    xywhn_boxes = tracking_results[0].boxes.xywhn.cpu().numpy().astype(int)
    xyxy_boxes = tracking_results[0].boxes.xyxy.cpu().numpy().astype(int)

    # Create additional variables to store useful info from the detections
    boxes_centers = xywh_boxes[:, :2]
    normalized_boxes_centers = xywhn_boxes[:, :2]
    boxes_corner1 = xyxy_boxes[:, :2]
    boxes_corner2 = xyxy_boxes[:, 2:]

    # Parse tracking ID
    if tracking_results[0].boxes.id is not None:
        ids_list = tracking_results[0].boxes.id.int().cpu().tolist()
        positions_list = normalized_boxes_centers.tolist()
    else:   # TODO is no tracking = no detections? (no)
        ids_list = []
        positions_list = []

    history.update(ids_list, positions_list)

    return classes, boxes_centers, normalized_boxes_centers, boxes_corner1, boxes_corner2


def perform_anomaly_detection(anomaly_detector, normalized_boxes_centers, anomaly_detection_args):
    pass


def send_alert(alerts_file, frame_id: int):
    # Write alert to file
    alerts_file.write(f"Alert: Frame {frame_id} - Anomalous behaviour detected.\n")


def draw_detections(
        annotated_frame,
        classes,
        are_anomalous,
        boxes_corner1,
        boxes_corner2,
):
    # drawing safety circles & detection boxes
    for obj_class, is_anomaly, box_corner1, box_corner2 in zip(classes, are_anomalous, boxes_corner1, boxes_corner2):
        # Choose color depending on class (blue sheep, purple goat) and it being an anomaly (red)
        color = RED if is_anomaly else CLASS_COLOR[obj_class]
        # Draw bounding box on frame
        cv2.rectangle(annotated_frame, box_corner1, box_corner2, color, 2)



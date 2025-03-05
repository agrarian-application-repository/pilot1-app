import numpy as np


class FrameQueueObject:

    def __init__(self, frame_id: int, frame: np.ndarray):
        self.frame_id = frame_id
        self.frame = frame


class DetectionResult:
    def __init__(
            self,
            frame_id: int,
            frame: np.ndarray,
            classes_names: list,
            num_classes: int,
            classes: np.ndarray,
            boxes_centers: np.ndarray,
            boxes_corner1: np.ndarray,
            boxes_corner2: np.ndarray,
    ):
        self.frame_id = frame_id
        self.frame = frame
        self.classes_names = classes_names
        self.num_classes = num_classes
        self.classes = classes
        self.boxes_centers = boxes_centers
        self.boxes_corner1 = boxes_corner1
        self.boxes_corner2 = boxes_corner2


class SegmentationResult:
    def __init__(
            self,
            frame_id: int,
            mask: np.ndarray,
    ):
        self.frame_id = frame_id
        self.mask = mask


class GeoResult:

    def __init__(
            self,
            frame_id: int,
            safety_radius_pixels: int,
            nodata_dem_mask: np.ndarray,
            geofencing_mask: np.ndarray,
            slope_mask: np.ndarray,
    ):
        self.frame_id = frame_id

        self.safety_radius_pixels = safety_radius_pixels
        self.nodata_dem_mask = nodata_dem_mask
        self.geofencing_mask = geofencing_mask
        self.slope_mask = slope_mask


class DangerDetectionResults:

    def __init__(
            self,
            frame_id: int,
            frame: np.ndarray,
            classes_names: list,
            num_classes: int,
            classes: np.ndarray,
            boxes_centers: np.ndarray,
            boxes_corner1: np.ndarray,
            boxes_corner2: np.ndarray,
            safety_radius_pixels: int,
            danger_mask: np.ndarray,
            intersection_mask: np.ndarray,
            danger_types: str,
    ):
        self.frame_id = frame_id
        self.frame = frame
        self.classes_names = classes_names
        self.num_classes = num_classes
        self.classes = classes
        self.boxes_centers = boxes_centers
        self.boxes_corner1 = boxes_corner1
        self.boxes_corner2 = boxes_corner2
        self.safety_radius_pixels = safety_radius_pixels
        self.danger_mask = danger_mask
        self.intersection_mask = intersection_mask
        self.danger_types = danger_types


class AnnotationResults:
    def __init__(
            self,
            frame_id: int,
            annotated_frame: np.ndarray,
            danger_types: str,
    ):
        self.frame_id = frame_id
        self.annotated_frame = annotated_frame
        self.danger_types = danger_types


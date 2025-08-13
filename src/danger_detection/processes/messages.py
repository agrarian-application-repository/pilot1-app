import numpy as np
from dataclasses import dataclass

@dataclass
class DetectionResult:
    """
    A dataclass to store the results of an object detection task.

    Attributes:
        frame_id (int): The unique identifier of the frame where detection occurred.
        frame (np.ndarray): The image data of the frame.
        classes_names (list[str]): A list of the names of the detected classes.
        num_classes (int): The number of unique classes detected.
        classes (np.ndarray): An array of class IDs for each detected object.
        boxes_centers (np.ndarray): An array of the center coordinates of each bounding box.
        boxes_corner1 (np.ndarray): An array of the top-left corner coordinates of each bounding box.
        boxes_corner2 (np.ndarray): An array of the bottom-right corner coordinates of each bounding box.
        timestamp (float): The timestamp of reception (of the frame).
    """
    frame_id: int
    frame: np.ndarray
    classes_names: list[str]
    num_classes: int
    classes: np.ndarray
    boxes_centers: np.ndarray
    boxes_corner1: np.ndarray
    boxes_corner2: np.ndarray
    timestamp: float


@dataclass
class SegmentationResult:
    """
    A dataclass to store the results of a segmentation task.

    Attributes:
        frame_id (int): The unique identifier of the frame.
        mask (np.ndarray): The segmentation mask as a NumPy array.
    """
    frame_id: int
    roads_mask: np.ndarray
    vehicles_mask: np.ndarray


@dataclass
class GeoResult:
    """
    A dataclass to store geographical and geometric analysis results.

    Attributes:
        frame_id (int): The unique identifier of the frame.
        safety_radius_pixels (int): The defined safety radius in pixels.
        nodata_dem_mask (np.ndarray): A mask for areas with no digital elevation model data.
        geofencing_mask (np.ndarray): A mask for geofenced areas.
        slope_mask (np.ndarray): A mask representing the slope analysis of the terrain.
    """
    frame_id: int
    safety_radius_pixels: int
    nodata_dem_mask: np.ndarray
    geofencing_mask: np.ndarray
    slope_mask: np.ndarray


@dataclass
class DangerDetectionResults:
    """
    A dataclass combining detection and geographical data to highlight danger areas.

    Attributes:
        frame_id (int): The unique identifier of the frame.
        frame (np.ndarray): The image data of the frame.
        classes_names (list[str]): A list of the names of the detected classes.
        num_classes (int): The number of unique classes detected.
        classes (np.ndarray): An array of class IDs for each detected object.
        boxes_centers (np.ndarray): An array of the center coordinates of each bounding box.
        boxes_corner1 (np.ndarray): An array of the top-left corner coordinates of each bounding box.
        boxes_corner2 (np.ndarray): An array of the bottom-right corner coordinates of each bounding box.
        safety_radius_pixels (int): The defined safety radius in pixels.
        danger_mask (np.ndarray): A mask highlighting areas of danger.
        intersection_mask (np.ndarray): A mask showing the intersection of detection and geo-analysis.
        danger_types (str): A list of string describing the types of danger detected.
        timestamp (float): The timestamp of reception (of the frame).
    """
    frame_id: int
    frame: np.ndarray
    classes_names: list[str]
    num_classes: int
    classes: np.ndarray
    boxes_centers: np.ndarray
    boxes_corner1: np.ndarray
    boxes_corner2: np.ndarray
    safety_radius_pixels: int
    danger_mask: np.ndarray
    intersection_mask: np.ndarray
    danger_types: str
    timestamp: float

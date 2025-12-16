import numpy as np
from dataclasses import dataclass


@dataclass
class FrameQueueObject:
    """
    A dataclass to represent a frame and its ID in a queue.

    Attributes:
        frame_id (int): The unique identifier for the frame.
        frame (np.ndarray): The actual image data as a NumPy array.
        timestamp (float): The timestamp of reception.
        original_wh tuple(int, int): The original shape of the image
    """
    frame_id: int
    frame: np.ndarray
    timestamp: float
    original_wh: tuple[int, int]


@dataclass
class TelemetryQueueObject:
    """
    A dataclass to represent a telemetry packet and its reception timestamp in a queue.

    Attributes:
        telemetry (dict): A dictionary object containing the drone telemetry.
        timestamp (float): The timestamp of reception.
    """
    telemetry: dict
    timestamp: float


@dataclass
class CombinedFrameTelemetryQueueObject:
    """
    A dataclass to represent the match between a frame and a telemetry packet based on timestamp.

    Attributes:
        frame_id (int): The unique identifier for the frame.
        frame (np.ndarray): The actual image data as a NumPy array.
        telemetry (dict|None): A dictionary object containing the drone telemetry.
        timestamp (float): The timestamp of reception (of the frame).
        original_wh tuple(int, int): The original shape of the image
    """
    frame_id: int
    frame: np.ndarray
    telemetry: dict|None
    timestamp: float
    original_wh: tuple[int, int]


@dataclass
class AnnotationResults:
    """
    A dataclass to store the final annotated frame and danger information.

    Attributes:
        frame_id (int): The unique identifier of the frame.
        annotated_frame (np.ndarray): The final image with annotations.
        alert_msg (str): A string describing the types of danger detected.
        timestamp (float): The timestamp of reception (of the frame).
        original_wh tuple(int, int): The original shape of the image
    """
    frame_id: int
    annotated_frame: np.ndarray
    alert_msg: str
    timestamp: float
    original_wh: tuple[int, int]

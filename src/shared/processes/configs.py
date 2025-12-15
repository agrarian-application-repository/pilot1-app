import numpy as np
from dataclasses import dataclass

@dataclass
class StreamVideoReaderConfig:
    """
    A dataclass to represent a frame and its ID in a queue.

    Attributes:
        frame_id (int): The unique identifier for the frame.
        frame (np.ndarray): The actual image data as a NumPy array.
        timestamp (float): The timestamp of reception.
    """
    frame_id: int
    frame: np.ndarray
    timestamp: float
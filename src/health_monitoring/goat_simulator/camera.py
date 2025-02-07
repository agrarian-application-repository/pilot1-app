import pygame
import random
from typing import Tuple

from constants import *


class Camera:
    """
    Represents a camera that moves and rotates.
    """
    def __init__(self, position: Tuple[int, int, int], speed: int, rotation_speed: int, aspect_ratio: float) -> None:
        """
        Initializes the camera.

        :param position: (x, y, z) position of the camera
        :param speed: Movement speed
        :param rotation_speed: Rotation speed
        :param aspect_ratio: Camera's aspect ratio
        """
        self.x, self.y, self.z = position
        self.speed = speed
        self.rotation_speed = rotation_speed
        self.aspect_ratio = aspect_ratio
        self.rotation = 0  # Camera initially looking straight down

    def move(self, dx: int, dy: int, dz: int) -> None:
        """
        Moves the camera while ensuring it stays within the terrain boundaries.
        """
        self.x += dx * self.speed
        self.y += dy * self.speed
        self.z = max(CAMERA_MIN_HEIGHT, min(self.z + dz * self.speed, CAMERA_MAX_HEIGHT))

    def rotate(self, d_rotation: int) -> None:
        """
        Rotates the camera.
        """
        self.rotation += d_rotation * self.rotation_speed

    def random_movement(self, move_prob: float, rotate_prob: float, elevation_prob: float) -> None:
        """
        Performs movement, rotation, and elevation changes based on probabilities.
        """
        if random.random() < move_prob:
            dx, dy = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            self.move(dx, dy, 0)
        if random.random() < rotate_prob:
            self.rotate(random.choice([-1, 1]))
        if random.random() < elevation_prob:
            self.move(0, 0, random.choice([-1, 1]))

    def get_viewport(self) -> Tuple[int, int, int, int]:
        """
        Returns the portion of the terrain visible to the camera.
        """
        view_width = int(WIDTH / self.aspect_ratio)
        view_height = HEIGHT
        return self.x, self.y, view_width, view_height



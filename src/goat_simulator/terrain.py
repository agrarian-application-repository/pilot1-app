import pygame
from typing import Tuple
from constants import *


class Terrain:
    """
    Represents a flat terrain.
    """

    def __init__(self, extent: Tuple[int, int], infinite: bool = False) -> None:
        """
        Initializes the terrain.

        :param extent: (width, height) of the terrain
        :param infinite: If True, the terrain will scroll infinitely
        """
        self.width, self.height = extent
        self.infinite = infinite
        self.color = (34, 139, 34)  # Green color
        self.offset = 0  # For infinite scrolling

    def draw(self, screen: pygame.Surface) -> None:
        """
        Draws the terrain on the screen.

        :param screen: The pygame screen surface
        """
        if self.infinite:
            self.offset = (self.offset + SCROLL_SPEED) % self.width
            screen.fill(self.color, (self.offset - self.width, HEIGHT - self.height, self.width, self.height))
        screen.fill(self.color, (0, HEIGHT - self.height, self.width, self.height))

    def get_surface_y(self) -> int:
        """
        Returns the y-coordinate of the top surface of the terrain.

        :return: Y-coordinate of the terrain surface
        """
        return HEIGHT - self.height
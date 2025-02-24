import pygame
from typing import Tuple, List
from constants import *


class Animal:
    """
    Represents an animal entity on the terrain.
    """
    def __init__(self, animal_id: int, position: Tuple[int, int], color: Tuple[int, int, int], speed: int) -> None:
        """
        Initializes an animal.

        :param animal_id: Unique identifier for the animal
        :param position: (x, y) position of the animal
        :param color: RGB color of the animal
        :param speed: Movement speed
        """
        self.animal_id = animal_id
        self.x, self.y = position
        self.color = color
        self.speed = speed
        self.rect = pygame.Rect(self.x, self.y, *ANIMAL_SIZE)

    def move(self, dx: int, dy: int, animals: List['Animal']) -> None:
        """
        Moves the animal while preventing collision with other animals.
        """
        new_rect = self.rect.move(dx * self.speed, dy * self.speed)
        if not any(new_rect.colliderect(animal.rect) for animal in animals if animal.animal_id != self.animal_id):
            self.rect = new_rect
            self.x, self.y = self.rect.topleft

    def draw(self, screen: pygame.Surface) -> None:
        """
        Draws the animal on the screen.
        """
        pygame.draw.rect(screen, self.color, self.rect)
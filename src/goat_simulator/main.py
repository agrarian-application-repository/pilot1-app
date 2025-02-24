import pygame
from typing import Tuple
from terrain import Terrain
from camera import Camera
from constants import *


def main() -> None:
    """
    Main function to run the pygame loop.
    """
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Terrain Example")
    clock = pygame.time.Clock()
    running = True

    terrain = Terrain((WIDTH, TERRAIN_HEIGHT), infinite=True)
    camera = Camera((WIDTH // 2, HEIGHT // 2, 150), speed=5, rotation_speed=2, aspect_ratio=16/9)

    while running:
        screen.fill((135, 206, 250))  # Sky blue background

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        terrain.draw(screen)
        camera.random_movement(0.1, 0.05, 0.05)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()


def main() -> None:
    """
    Main function to run the pygame loop.
    """
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Terrain Example")
    clock = pygame.time.Clock()
    running = True

    terrain = Terrain((WIDTH, TERRAIN_HEIGHT), infinite=True)
    animals = [Animal(i, (random.randint(0, WIDTH), terrain.get_surface_y() - ANIMAL_SIZE[1]), (255, 0, 0), 2) for i in
               range(5)]

    while running:
        screen.fill((135, 206, 250))  # Sky blue background

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        terrain.draw(screen)
        for animal in animals:
            animal.move(random.choice([-1, 1]), 0, animals)  # Move left or right
            animal.draw(screen)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
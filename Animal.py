import pygame
from pygame.locals import *
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class Animal(ABC):
    x: float = 0.0
    y: float = 0.0
    direction: float = 0.0
    resources: float = 0.0
    eating_distance: float = 50.0

    def move_forward(self, distance, grid_boundaries_x, grid_boundaries_y):
        new_x = self.x + distance * np.cos(np.radians(self.direction))
        new_y = self.y + distance * np.sin(np.radians(self.direction))

        if -grid_boundaries_x <= new_x <= grid_boundaries_x and -grid_boundaries_y <= new_y <= grid_boundaries_y:
            self.x = new_x
            self.y = new_y

    def turn(self, angle_degrees):
        self.direction = (self.direction + angle_degrees) % 360

    @abstractmethod
    def eat(self, food):
        pass

    def plot(self, screen):
        # Plot the animal and its direction on the Pygame screen
        pygame.draw.circle(screen, (255, 0, 0), (int(self.x), int(self.y)), 10)  # Animal's position
        direction_vector = (self.x + 20 * np.cos(np.radians(self.direction)), self.y + 20 * np.sin(np.radians(self.direction)))
        pygame.draw.line(screen, (0, 0, 255), (int(self.x), int(self.y)), (int(direction_vector[0]), int(direction_vector[1])), 3)
        # Print the animal's resources as text
        font = pygame.font.Font(None, 36)
        text = font.render(f"Resources: {self.resources}", True, (255, 255, 255))
        screen.blit(text, (10, 10))
        # plot a circle around the animal to indicate the distance at which it can eat
        pygame.draw.circle(screen, (255, 255, 255), (int(self.x), int(self.y)), self.eating_distance, 1)


@dataclass
class Plant:
    x: float = 0.0
    y: float = 0.0
    resources: float = 0.0

    def plot(self, screen):
        pygame.draw.circle(screen, (0, 255, 0), (int(self.x), int(self.y)), 10)  # Plant's position
        # Print the plant's resources as text
        font = pygame.font.Font(None, 36)
        text = font.render(f"Resources: {self.resources}", True, (255, 255, 255))
        screen.blit(text, (self.x + 20, self.y - 20))

@dataclass
class Herbivore(Animal):

    def eat(self, food: Plant):
            amount_to_eat = food.resources * 0.10
            self.resources += amount_to_eat
            food.resources -= amount_to_eat

@dataclass
class World:
    animals: list[Animal]
    plants: list[Plant]
    grid_boundaries_x: float
    grid_boundaries_y: float

    def create_herbivore(self, x, y):
        animal = Herbivore(x, y)
        self.animals.append(animal)

    def create_plant(self, x, y, resources):
        plant = Plant(x, y, resources)
        self.plants.append(plant)

    def can_eat(self, animal: Herbivore, plant: Plant):
        return  np.sqrt((animal.x - plant.x) ** 2 + (animal.y - plant.y) ** 2) < animal.eating_distance

    def eat_plant(self, animal: Herbivore, plant: Plant):
        if self.can_eat(animal, plant):
            animal.eat(plant)
            return True
        return False

    def plot(self, screen):
        for animal in self.animals:
            animal.plot(screen)
        for plant in self.plants:
            plant.plot(screen)

    def simulate(self, screen):
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            if keys[K_UP]:
                for animal in self.animals:
                    animal.move_forward(1.0, self.grid_boundaries_x, self.grid_boundaries_y)
            if keys[K_RIGHT]:
                for animal in self.animals:
                    animal.turn(5.0)
            if keys[K_LEFT]:
                for animal in self.animals:
                    animal.turn(-5.0)
            if keys[K_SPACE]:
                for animal in self.animals:
                    for plant in self.plants:
                        print(f'can eat: {self.can_eat(animal, plant)} iterationclock {clock.get_time()}')
                        if self.can_eat(animal, plant):
                            self.eat_plant(animal, plant
                                             )
                            break


            screen.fill((0, 0, 0))
            self.plot(screen)
            pygame.display.flip()
            clock.tick(30)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))

world = World(grid_boundaries_x=800, grid_boundaries_y=600, animals=[], plants=[])
world.create_herbivore(300, 300)
world.create_plant(400, 300, 10)

world.simulate(screen)

pygame.quit()

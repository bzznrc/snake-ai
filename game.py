import pygame
import random
from enum import Enum
from collections import namedtuple
from constants import *

# Initialize Pygame
pygame.init()
#font = pygame.font.SysFont('Segoe UI', 20)
font = pygame.font.SysFont(None, 24)

# Direction enumeration for movement
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Point to represent positions on the grid
Point = namedtuple('Point', 'x, y')

class Game:
    """Base class representing the Game."""

    def __init__(self):
        """Initialize the game state."""
        self.w = SCREEN_WIDTH
        self.h = SCREEN_HEIGHT
        # Initialize display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        """Reset the game state for a new episode."""
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2 - BB_HEIGHT // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]
        self.score = 0
        self.food = None
        self.obstacles = []
        self._place_food()
        self._place_obstacles()
        self.frame_iteration = 0  # Count steps to prevent looping

    def _place_food(self):
        """Place food at a random location not occupied by the snake or obstacles."""
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BB_HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            if y >= self.h - BB_HEIGHT:
                continue
            self.food = Point(x, y)
            if self.food not in self.snake and self.food not in self.obstacles:
                break

    def _place_obstacles(self):
        """Place multiple obstacles (sections) in random locations."""
        self.obstacles = []
        for _ in range(NUM_OBSTACLES):
            num_sections = random.randint(MIN_SECTIONS, MAX_SECTIONS)
            # Random starting point
            attempts = 0
            max_attempts = 100
            while attempts < max_attempts:
                x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                y = random.randint(0, (self.h - BB_HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                if y >= self.h - BB_HEIGHT:
                    attempts += 1
                    continue
                start_point = Point(x, y)
                if start_point in self.snake or start_point == self.food or start_point in self.obstacles:
                    attempts += 1
                    continue
                break
            else:
                # Failed to place this obstacle, skip
                continue

            # Generate a connected open shape
            directions = [(-BLOCK_SIZE, 0), (BLOCK_SIZE, 0), (0, -BLOCK_SIZE), (0, BLOCK_SIZE)]
            shape = [start_point]
            current_point = start_point

            for _ in range(num_sections - 1):
                random.shuffle(directions)
                for dx, dy in directions:
                    next_point = Point(current_point.x + dx, current_point.y + dy)
                    # Check boundaries and overlaps
                    if (0 <= next_point.x < self.w and
                        0 <= next_point.y < self.h - BB_HEIGHT and
                        next_point not in self.snake and
                        next_point != self.food and
                        next_point not in self.obstacles and
                        next_point not in shape):
                        shape.append(next_point)
                        current_point = next_point
                        break
                else:
                    # Cannot find a valid extension, stop building the shape
                    break

            self.obstacles.extend(shape)

    def _update_ui(self):
        """Update the game's UI."""
        self.display.fill(COLOR_BACKGROUND)

        # Draw the snake
        for pt in self.snake:
            # Draw outline
            pygame.draw.rect(self.display, COLOR_SNAKE_OUTLINE,
                             pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            # Draw fill
            pygame.draw.rect(self.display, COLOR_SNAKE_PRIMARY,
                             pygame.Rect(pt.x + 4, pt.y + 4, BLOCK_SIZE - 8, BLOCK_SIZE - 8))

        # Draw the food
        pygame.draw.rect(self.display, COLOR_FRUIT_OUTLINE,
                         pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, COLOR_FRUIT_PRIMARY,
                         pygame.Rect(self.food.x + 4, self.food.y + 4, BLOCK_SIZE - 8, BLOCK_SIZE - 8))

        # Draw the obstacles (walls)
        for pt in self.obstacles:
            # Draw outline
            pygame.draw.rect(self.display, COLOR_OBSTACLE_OUTLINE,
                             pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            # Draw fill
            pygame.draw.rect(self.display, COLOR_OBSTACLE_PRIMARY,
                             pygame.Rect(pt.x + 4, pt.y + 4, BLOCK_SIZE - 8, BLOCK_SIZE - 8))

        # Draw the bottom bar
        pygame.draw.rect(self.display, (0, 0, 0),
                         pygame.Rect(0, self.h - BB_HEIGHT, self.w, BB_HEIGHT))

        # Display the score on the bottom bar
        score_text = font.render(f"Score: {self.score}", True, COLOR_SCORE)
        score_rect = score_text.get_rect(topleft=(BB_MARGIN, self.h - BB_HEIGHT + (BB_HEIGHT - score_text.get_height()) // 2))
        self.display.blit(score_text, score_rect)

        pygame.display.flip()

    def _move(self, direction):
        """Update the head position based on the current direction."""
        # To be implemented in subclasses
        pass

    def is_collision(self, pt=None):
        """Check if the snake has collided with itself or an obstacle."""
        if pt is None:
            pt = self.head
        # Collision with self
        if pt in self.snake[1:]:
            return True
        # Collision with obstacles
        if pt in self.obstacles:
            return True
        return False

    def _handle_wall_collision(self):
        """Handle snake passing through walls if WRAP_AROUND is True."""
        x = self.head.x
        y = self.head.y

        if WRAP_AROUND:
            if x >= self.w:
                x = 0
            elif x < 0:
                x = self.w - BLOCK_SIZE
            if y >= self.h - BB_HEIGHT:
                y = 0
            elif y < 0:
                y = self.h - BB_HEIGHT - BLOCK_SIZE

            self.head = Point(x, y)
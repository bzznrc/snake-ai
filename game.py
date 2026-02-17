import pygame
import random
from enum import Enum
from collections import namedtuple

from config import (
    BB_HEIGHT,
    CELL_INSET,
    MAX_OBSTACLE_SECTIONS,
    MIN_OBSTACLE_SECTIONS,
    NUM_OBSTACLES,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    TILE_SIZE,
    UI_STATUS_SEPARATOR,
    WINDOW_TITLE,
    WRAP_AROUND,
)
from bgds.boards.square_generation import spawn_connected_random_walk_shapes
from bgds.visual.assets import load_font
from bgds.visual.colors import (
    COLOR_AQUA,
    COLOR_BRICK_RED,
    COLOR_CHARCOAL,
    COLOR_CORAL,
    COLOR_DEEP_TEAL,
    COLOR_NEAR_BLACK,
    COLOR_SLATE_GRAY,
    COLOR_SOFT_WHITE,
)
from bgds.visual.square_render import draw_two_tone_cell, draw_two_tone_grid_block
from bgds.visual.statusbar import draw_centered_status_bar
from bgds.visual.typography import FONT_FAMILY_DEFAULT, FONT_PATH_ROBOTO_REGULAR

# Initialize Pygame
pygame.init()
font = load_font(
    FONT_PATH_ROBOTO_REGULAR,
    24,
    fallback_family=FONT_FAMILY_DEFAULT,
)

# Direction enumeration for movement
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Point to represent positions on the grid
Point = namedtuple('Point', 'x, y')

class BaseGame:
    """Base class representing the game world and shared rendering."""

    def __init__(self):
        """Initialize the game state."""
        self.width = SCREEN_WIDTH
        self.height = SCREEN_HEIGHT
        # Initialize display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(WINDOW_TITLE)
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        """Reset the game state for a new episode."""
        self.direction = Direction.RIGHT
        self.head = Point(self.width / 2, self.height / 2 - BB_HEIGHT // 2)
        self.snake = [
            self.head,
            Point(self.head.x - TILE_SIZE, self.head.y),
            Point(self.head.x - (2 * TILE_SIZE), self.head.y),
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
            x = random.randint(0, (self.width - TILE_SIZE) // TILE_SIZE) * TILE_SIZE
            y = random.randint(0, (self.height - BB_HEIGHT - TILE_SIZE) // TILE_SIZE) * TILE_SIZE
            if y >= self.height - BB_HEIGHT:
                continue
            self.food = Point(x, y)
            if self.food not in self.snake and self.food not in self.obstacles:
                break

    def _place_obstacles(self):
        """Place multiple obstacles (sections) in random locations."""
        self.obstacles = []
        shapes = spawn_connected_random_walk_shapes(
            shape_count=NUM_OBSTACLES,
            min_sections=MIN_OBSTACLE_SECTIONS,
            max_sections=MAX_OBSTACLE_SECTIONS,
            sample_start_fn=self._sample_valid_obstacle_start,
            neighbor_candidates_fn=self._neighbor_obstacle_candidates,
            is_candidate_valid_fn=self._is_valid_obstacle_tile,
        )
        for shape in shapes:
            self.obstacles.extend(shape)

    def _sample_valid_obstacle_start(self):
        """Sample a valid start tile for an obstacle shape."""
        for _ in range(100):
            x = random.randint(0, (self.width - TILE_SIZE) // TILE_SIZE) * TILE_SIZE
            y = random.randint(0, (self.height - BB_HEIGHT - TILE_SIZE) // TILE_SIZE) * TILE_SIZE
            point = Point(x, y)
            if self._is_valid_obstacle_tile(point, []):
                return point
        return None

    @staticmethod
    def _neighbor_obstacle_candidates(point):
        """Return four-neighbor candidate tiles from a source tile."""
        return [
            Point(point.x - TILE_SIZE, point.y),
            Point(point.x + TILE_SIZE, point.y),
            Point(point.x, point.y - TILE_SIZE),
            Point(point.x, point.y + TILE_SIZE),
        ]

    def _is_valid_obstacle_tile(self, tile, pending_tiles):
        """Validate obstacle placement against bounds and existing occupancy."""
        if not (0 <= tile.x < self.width and 0 <= tile.y < self.height - BB_HEIGHT):
            return False
        if tile in self.snake or tile == self.food:
            return False
        if tile in self.obstacles or tile in pending_tiles:
            return False
        return True

    def _update_ui(self):
        """Update the game's UI."""
        self.display.fill(COLOR_CHARCOAL)

        # Draw the snake
        draw_two_tone_grid_block(
            surface=self.display,
            top_left_points=self.snake,
            size_px=TILE_SIZE,
            inset_px=CELL_INSET,
            outer_color=COLOR_AQUA,
            inner_color=COLOR_DEEP_TEAL,
        )

        # Draw the food
        draw_two_tone_cell(
            surface=self.display,
            top_left=self.food,
            size_px=TILE_SIZE,
            inset_px=CELL_INSET,
            outer_color=COLOR_CORAL,
            inner_color=COLOR_BRICK_RED,
        )

        # Draw the obstacles (walls)
        draw_two_tone_grid_block(
            surface=self.display,
            top_left_points=self.obstacles,
            size_px=TILE_SIZE,
            inset_px=CELL_INSET,
            outer_color=COLOR_SLATE_GRAY,
            inner_color=COLOR_CHARCOAL,
        )

        draw_centered_status_bar(
            surface=self.display,
            font=font,
            screen_width_px=self.width,
            screen_height_px=self.height,
            bar_height_px=BB_HEIGHT,
            items=[f"Score: {self.score}"],
            background_color=COLOR_NEAR_BLACK,
            default_text_color=COLOR_SOFT_WHITE,
            separator=UI_STATUS_SEPARATOR,
            separator_color=COLOR_SOFT_WHITE,
        )

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
            if x >= self.width:
                x = 0
            elif x < 0:
                x = self.width - TILE_SIZE
            if y >= self.height - BB_HEIGHT:
                y = 0
            elif y < 0:
                y = self.height - BB_HEIGHT - TILE_SIZE

            self.head = Point(x, y)


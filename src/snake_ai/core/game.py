"""Base Snake game world and rendering logic."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import random

import arcade

from snake_ai.boards import spawn_connected_random_walk_shapes
from snake_ai.config import (
    BB_HEIGHT,
    CELL_INSET,
    MAX_OBSTACLE_SECTIONS,
    MIN_OBSTACLE_SECTIONS,
    NUM_OBSTACLES,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    SHOW_GAME,
    TILE_SIZE,
    WINDOW_TITLE,
    WRAP_AROUND,
)
from snake_ai.runtime import ArcadeFrameClock, ArcadeWindowController, TextCache, load_font_once
from snake_ai.visual import (
    COLOR_AQUA,
    COLOR_BRICK_RED,
    COLOR_CHARCOAL,
    COLOR_CORAL,
    COLOR_DEEP_TEAL,
    COLOR_NEAR_BLACK,
    COLOR_SLATE_GRAY,
    COLOR_SOFT_WHITE,
    FONT_FAMILY_DEFAULT,
    FONT_PATH_ROBOTO_REGULAR,
    resolve_font_path,
)


def _font_name() -> str:
    font_path = resolve_font_path(FONT_PATH_ROBOTO_REGULAR)
    load_font_once(font_path)
    return FONT_FAMILY_DEFAULT or "Roboto"


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


@dataclass(frozen=True)
class Point:
    x: float
    y: float


class BaseGame:
    """Base class representing the game world and shared rendering."""

    def __init__(self):
        self.width = SCREEN_WIDTH
        self.height = SCREEN_HEIGHT
        self.frame_clock = ArcadeFrameClock()
        self.font_name = _font_name()
        self.text_cache = TextCache(max_entries=64)
        self.window_controller = ArcadeWindowController(
            self.width,
            self.height,
            WINDOW_TITLE,
            enabled=SHOW_GAME,
            queue_input_events=False,
            vsync=False,
        )
        self.window = self.window_controller.window

        self.reset()

    def close(self):
        self.window_controller.close()
        self.window = None

    def poll_events(self):
        self.window_controller.poll_events_or_raise()

    def _poll_window_events(self):
        self.poll_events()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.width / 2, self.height / 2 - BB_HEIGHT // 2)
        self.snake = [
            self.head,
            Point(self.head.x - TILE_SIZE, self.head.y),
            Point(self.head.x - (2 * TILE_SIZE), self.head.y),
        ]
        self.score = 0
        self.food = None
        self.obstacles: list[Point] = []
        self._place_food()
        self._place_obstacles()
        self.frame_iteration = 0

    def _place_food(self):
        while True:
            x = random.randint(0, (self.width - TILE_SIZE) // TILE_SIZE) * TILE_SIZE
            y = random.randint(0, (self.height - BB_HEIGHT - TILE_SIZE) // TILE_SIZE) * TILE_SIZE
            if y >= self.height - BB_HEIGHT:
                continue
            self.food = Point(x, y)
            if self.food not in self.snake and self.food not in self.obstacles:
                break

    def _place_obstacles(self):
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
        for _ in range(100):
            x = random.randint(0, (self.width - TILE_SIZE) // TILE_SIZE) * TILE_SIZE
            y = random.randint(0, (self.height - BB_HEIGHT - TILE_SIZE) // TILE_SIZE) * TILE_SIZE
            point = Point(x, y)
            if self._is_valid_obstacle_tile(point, []):
                return point
        return None

    @staticmethod
    def _neighbor_obstacle_candidates(point):
        return [
            Point(point.x - TILE_SIZE, point.y),
            Point(point.x + TILE_SIZE, point.y),
            Point(point.x, point.y - TILE_SIZE),
            Point(point.x, point.y + TILE_SIZE),
        ]

    def _is_valid_obstacle_tile(self, tile, pending_tiles):
        if not (0 <= tile.x < self.width and 0 <= tile.y < self.height - BB_HEIGHT):
            return False
        if tile in self.snake or tile == self.food:
            return False
        if tile in self.obstacles or tile in pending_tiles:
            return False
        return True

    def _to_window_y(self, top_y: float, height: float = 0) -> float:
        return self.window_controller.to_arcade_y(float(top_y) + float(height))

    def _draw_tile(self, top_left, outer_color, inner_color):
        x = float(top_left.x)
        y = float(top_left.y)
        bottom = self._to_window_y(y, TILE_SIZE)

        arcade.draw_lbwh_rectangle_filled(x, bottom, TILE_SIZE, TILE_SIZE, outer_color)
        inner_size = TILE_SIZE - 2 * CELL_INSET
        if inner_size > 0:
            arcade.draw_lbwh_rectangle_filled(
                x + CELL_INSET,
                bottom + CELL_INSET,
                inner_size,
                inner_size,
                inner_color,
            )

    def _draw_tile_batch(self, tiles, outer_color, inner_color):
        for tile in tiles:
            self._draw_tile(tile, outer_color, inner_color)

    def _draw_status_bar(self):
        arcade.draw_lbwh_rectangle_filled(0, 0, self.width, BB_HEIGHT, COLOR_NEAR_BLACK)
        self.text_cache.draw(
            text=f"Score: {self.score}",
            x=self.width * 0.5,
            y=BB_HEIGHT * 0.5,
            color=COLOR_SOFT_WHITE,
            font_size=24,
            font_name=self.font_name,
            anchor_x="center",
            anchor_y="center",
        )

    def draw_frame(self):
        if self.window is None:
            return

        self.window_controller.clear(COLOR_CHARCOAL)
        self._draw_tile_batch(self.snake, COLOR_AQUA, COLOR_DEEP_TEAL)
        self._draw_tile(self.food, COLOR_CORAL, COLOR_BRICK_RED)
        self._draw_tile_batch(self.obstacles, COLOR_SLATE_GRAY, COLOR_CHARCOAL)
        self._draw_status_bar()
        self.window_controller.flip()

    def _move(self, direction):
        raise NotImplementedError

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt in self.snake[1:]:
            return True
        if pt in self.obstacles:
            return True
        return False

    def _handle_wall_collision(self):
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

"""Training environment wrapping Snake gameplay with RL actions."""

import pygame
import numpy as np

from config import BB_HEIGHT, TILE_SIZE, TRAINING_FPS, WRAP_AROUND
from game import BaseGame, Direction, Point

class TrainingGame(BaseGame):
    """AI-controlled training environment."""

    def play_step(self, action):
        """Execute one game step based on the selected action."""
        self.frame_iteration += 1

        # Handle quit events during training loops.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        # Move snake and append new head.
        self._move(action)
        self.snake.insert(0, self.head)

        # End episode on collision or stalling.
        reward = 0
        game_over = False
        if self._has_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # Eat food or keep moving.
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
            self.frame_iteration = 0
        else:
            self.snake.pop()

        self._update_ui()
        self.clock.tick(TRAINING_FPS)

        return reward, game_over, self.score

    def get_state_vector(self):
        """Build the 11-feature state vector used by the Q-network."""
        head = self.snake[0]
        point_l = Point(head.x - TILE_SIZE, head.y)
        point_r = Point(head.x + TILE_SIZE, head.y)
        point_u = Point(head.x, head.y - TILE_SIZE)
        point_d = Point(head.x, head.y + TILE_SIZE)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            (dir_r and self.is_collision(point_r))
            or (dir_l and self.is_collision(point_l))
            or (dir_u and self.is_collision(point_u))
            or (dir_d and self.is_collision(point_d)),
            (dir_u and self.is_collision(point_r))
            or (dir_d and self.is_collision(point_l))
            or (dir_l and self.is_collision(point_u))
            or (dir_r and self.is_collision(point_d)),
            (dir_d and self.is_collision(point_r))
            or (dir_u and self.is_collision(point_l))
            or (dir_r and self.is_collision(point_u))
            or (dir_l and self.is_collision(point_d)),
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            self.food.x < self.head.x,
            self.food.x > self.head.x,
            self.food.y < self.head.y,
            self.food.y > self.head.y,
        ]

        return np.array(state, dtype=int)

    def _move(self, action):
        """Update the head position based on [straight, right, left] action."""
        clockwise_directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clockwise_directions.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clockwise_directions[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clockwise_directions[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clockwise_directions[next_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += TILE_SIZE
        elif self.direction == Direction.LEFT:
            x -= TILE_SIZE
        elif self.direction == Direction.DOWN:
            y += TILE_SIZE
        elif self.direction == Direction.UP:
            y -= TILE_SIZE
        self.head = Point(x, y)

        if WRAP_AROUND:
            self._handle_wall_collision()
        else:
            if x < 0 or x >= self.width or y < 0 or y >= self.height - BB_HEIGHT:
                self.head = Point(x, y)

    def _has_collision(self):
        """Check collision with walls (when enabled), self, or obstacles."""
        if not WRAP_AROUND:
            if (
                self.head.x < 0
                or self.head.x >= self.width
                or self.head.y < 0
                or self.head.y >= self.height - BB_HEIGHT
            ):
                return True
        return self.is_collision()


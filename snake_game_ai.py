from snake_game import SnakeGame, Direction, Point
import pygame
import numpy as np
from constants import *

class SnakeGameAI(SnakeGame):
    """AI-controlled version of the Snake game."""

    def play_step(self, action):
        """Execute one game step based on the action taken."""
        self.frame_iteration += 1

        # 1. Handle user input (exit game if needed)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. Move the snake
        self._move(action)
        self.snake.insert(0, self.head)

        # 3. Check if game over
        reward = 0
        game_over = False
        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. Check if food is eaten
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
            self.frame_iteration = 0  # Reset since we got food
        else:
            # Remove the last segment of the snake
            self.snake.pop()

        # 5. Update the UI and clock
        self._update_ui()
        self.clock.tick(AI_FPS)

        # 6. Return reward, game over flag, and score
        return reward, game_over, self.score

    def _move(self, action):
        """Update the head position based on the action taken."""
        # Actions: [straight, right turn, left turn]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            # Move straight
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            # Right turn
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:  # [0, 0, 1]
            # Left turn
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)

        # Handle wall collision
        if WRAP_AROUND:
            self._handle_wall_collision()
        else:
            if x < 0 or x >= self.w or y < 0 or y >= self.h - BB_HEIGHT:
                self.head = Point(x, y)  # Keep the position to trigger collision

    def _is_collision(self):
        """Check for collision with walls, self, or obstacles."""
        # Collision with walls
        if not WRAP_AROUND:
            if (self.head.x < 0 or self.head.x >= self.w or
                self.head.y < 0 or self.head.y >= self.h - BB_HEIGHT):
                return True
        return self.is_collision()
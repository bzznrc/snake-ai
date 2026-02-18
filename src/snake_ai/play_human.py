"""Human-play loop for Snake."""

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import arcade

from snake_ai.config import BB_HEIGHT, FPS, TILE_SIZE, WRAP_AROUND
from snake_ai.core import BaseGame, Direction, Point


class HumanGame(BaseGame):
    """User-controlled Snake game mode."""

    def play_step(self):
        self.frame_iteration += 1
        self.poll_events()

        if self.window_controller.is_key_down(arcade.key.A) and self.direction != Direction.RIGHT:
            self.direction = Direction.LEFT
        elif self.window_controller.is_key_down(arcade.key.D) and self.direction != Direction.LEFT:
            self.direction = Direction.RIGHT
        elif self.window_controller.is_key_down(arcade.key.W) and self.direction != Direction.DOWN:
            self.direction = Direction.UP
        elif self.window_controller.is_key_down(arcade.key.S) and self.direction != Direction.UP:
            self.direction = Direction.DOWN

        self._move(self.direction)
        self.snake.insert(0, self.head)

        game_over = False
        if self._has_collision():
            game_over = True
            return game_over, self.score

        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        self.draw_frame()
        self.frame_clock.tick(FPS)

        return game_over, self.score

    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += TILE_SIZE
        elif direction == Direction.LEFT:
            x -= TILE_SIZE
        elif direction == Direction.DOWN:
            y += TILE_SIZE
        elif direction == Direction.UP:
            y -= TILE_SIZE
        self.head = Point(x, y)

        if WRAP_AROUND:
            self._handle_wall_collision()
        else:
            if x < 0 or x >= self.width or y < 0 or y >= self.height - BB_HEIGHT:
                self.head = Point(x, y)

    def _has_collision(self):
        if not WRAP_AROUND:
            if self.head.x < 0 or self.head.x >= self.width or self.head.y < 0 or self.head.y >= self.height - BB_HEIGHT:
                return True
        return self.is_collision()


def run_human():
    game = HumanGame()
    score = 0
    try:
        while True:
            game_over, score = game.play_step()
            if game_over:
                break
    finally:
        game.close()

    print("Final Score:", score)


if __name__ == "__main__":
    run_human()

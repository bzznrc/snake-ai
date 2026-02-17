"""Human-play loop for Snake."""

import pygame

from config import BB_HEIGHT, FPS, TILE_SIZE, WRAP_AROUND
from game import BaseGame, Direction, Point

class HumanGame(BaseGame):
    """User-controlled Snake game mode."""

    def play_step(self):
        """Execute one game step from keyboard input."""
        self.frame_iteration += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            elif event.type == pygame.KEYDOWN:
                # Prevent reversing direction.
                if event.key == pygame.K_a and self.direction != Direction.RIGHT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_d and self.direction != Direction.LEFT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_w and self.direction != Direction.DOWN:
                    self.direction = Direction.UP
                elif event.key == pygame.K_s and self.direction != Direction.UP:
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

        self._update_ui()
        self.clock.tick(FPS)

        return game_over, self.score

    def _move(self, direction):
        """Update head position based on current direction."""
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
        """Check collision with walls, self, or obstacles."""
        if not WRAP_AROUND:
            if (
                self.head.x < 0
                or self.head.x >= self.width
                or self.head.y < 0
                or self.head.y >= self.height - BB_HEIGHT
            ):
                return True
        return self.is_collision()

def play_human():
    """Run the human-play game loop."""
    game = HumanGame()

    while True:
        game_over, score = game.play_step()
        if game_over:
            break

    print("Final Score:", score)
    pygame.quit()

if __name__ == "__main__":
    play_human()


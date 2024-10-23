from game import Game, Direction, Point
import pygame
from constants import *

class GameUser(Game):
    """User-controlled version of the Snake game."""

    def play_step(self):
        """Execute one game step."""
        self.frame_iteration += 1

        # 1. Handle user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                # Prevent the snake from reversing direction
                if event.key == pygame.K_LEFT and self.direction != Direction.RIGHT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT and self.direction != Direction.LEFT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP and self.direction != Direction.DOWN:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN and self.direction != Direction.UP:
                    self.direction = Direction.DOWN

        # 2. Move the snake
        self._move(self.direction)
        self.snake.insert(0, self.head)

        # 3. Check if game over
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score

        # 4. Check if food is eaten
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            # Remove the last segment of the snake
            self.snake.pop()

        # 5. Update the UI and clock
        self._update_ui()
        self.clock.tick(FPS)

        # 6. Return game over and score
        return game_over, self.score

    def _move(self, direction):
        """Update the head position based on the current direction."""
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
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

if __name__ == '__main__':
    game = GameUser()

    # Game loop
    while True:
        game_over, score = game.play_step()

        if game_over:
            break

    print('Final Score:', score)
    pygame.quit()
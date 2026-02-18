"""Central configuration for Snake AI."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RuntimeConfig:
    show_game: bool


@dataclass(frozen=True)
class BoardConfig:
    columns: int
    rows: int
    cell_size_px: int
    bottom_bar_height_px: int
    cell_inset_px: int

    @property
    def screen_width_px(self) -> int:
        return self.columns * self.cell_size_px

    @property
    def screen_height_px(self) -> int:
        return self.rows * self.cell_size_px + self.bottom_bar_height_px


RUNTIME = RuntimeConfig(show_game=_env_flag("SNAKE_SHOW_GAME", True))
BOARD = BoardConfig(
    columns=32,
    rows=24,
    cell_size_px=20,
    bottom_bar_height_px=30,
    cell_inset_px=4,
)

# Runtime
SHOW_GAME = RUNTIME.show_game
LOAD_MODEL = True
FPS = 20
TRAINING_FPS = 0
WINDOW_TITLE = "Snake AI"

# Arena
GRID_WIDTH_TILES = BOARD.columns
GRID_HEIGHT_TILES = BOARD.rows
TILE_SIZE = BOARD.cell_size_px
BB_HEIGHT = BOARD.bottom_bar_height_px
SCREEN_WIDTH = BOARD.screen_width_px
SCREEN_HEIGHT = BOARD.screen_height_px
CELL_INSET = BOARD.cell_inset_px
CELL_INSET_DOUBLE = CELL_INSET * 2

# Gameplay
NUM_OBSTACLES = 10
MIN_OBSTACLE_SECTIONS = 2
MAX_OBSTACLE_SECTIONS = 5
WRAP_AROUND = True

# Model and training
MODEL_NAME = "snake_32"
MODEL_PATH = str(PROJECT_ROOT / "model" / f"{MODEL_NAME}.pth")
MAX_MEMORY = 100_000
BATCH_SIZE = 1_000
LEARNING_RATE = 1e-3
HIDDEN_DIMENSIONS = [32]
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_DECAY = 0.9995
EPSILON_MIN = 0.01

# UI
FONT_FAMILY_DEFAULT: str | None = None
FONT_PATH_ROBOTO_REGULAR = "fonts/Roboto-Regular.ttf"
FONT_SIZE_STATUS = 24
UI_STATUS_SEPARATOR = "   /   "

# Colors
COLOR_AQUA = (102, 212, 200)
COLOR_DEEP_TEAL = (38, 110, 105)
COLOR_CORAL = (244, 137, 120)
COLOR_BRICK_RED = (150, 62, 54)
COLOR_SLATE_GRAY = (97, 101, 107)
COLOR_CHARCOAL = (28, 30, 36)
COLOR_NEAR_BLACK = (18, 18, 22)
COLOR_SOFT_WHITE = (238, 238, 242)

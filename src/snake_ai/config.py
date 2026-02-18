"""Configuration values for the Snake RL project."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from snake_ai.boards import SQUARE_BOARD_STANDARD, SQUARE_CELL_RENDER_STANDARD
from snake_ai.visual import STATUS_SEPARATOR_SLASH


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RuntimeFlags:
    show_game: bool


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FLAGS = RuntimeFlags(show_game=_env_flag("SNAKE_SHOW_GAME", False))

# Quick toggles
SHOW_GAME = FLAGS.show_game
LOAD_MODEL = True

# Runtime
FPS = 20
TRAINING_FPS = False
WINDOW_TITLE = "Snake AI"

# Arena dimensions
GRID_WIDTH_TILES = SQUARE_BOARD_STANDARD.columns
GRID_HEIGHT_TILES = SQUARE_BOARD_STANDARD.rows
TILE_SIZE = SQUARE_BOARD_STANDARD.cell_size_px
BB_HEIGHT = SQUARE_BOARD_STANDARD.bottom_bar_height_px
SCREEN_WIDTH = SQUARE_BOARD_STANDARD.screen_width_px
SCREEN_HEIGHT = SQUARE_BOARD_STANDARD.screen_height_px
CELL_INSET = SQUARE_CELL_RENDER_STANDARD.inset_px
CELL_INSET_DOUBLE = SQUARE_CELL_RENDER_STANDARD.inset_double_px

# Rendering
UI_STATUS_SEPARATOR = STATUS_SEPARATOR_SLASH

# Gameplay tuning
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
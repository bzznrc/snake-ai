"""Configuration values for the Snake RL project."""

from pathlib import Path
import sys

# Prefer local project modules when multiple game folders are on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Allow local workspace runs without installing bgds globally.
_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
_BGDS_SRC = _WORKSPACE_ROOT / "bazza-game-design-system" / "src"
if _BGDS_SRC.exists() and str(_BGDS_SRC) not in sys.path:
    sys.path.insert(0, str(_BGDS_SRC))

from bgds.boards.square import SQUARE_BOARD_STANDARD, SQUARE_CELL_RENDER_STANDARD
from bgds.visual.typography import STATUS_SEPARATOR_SLASH

# Quick toggles
SHOW_GAME = True
PLOT_TRAINING = False
LOAD_MODEL = True

# Runtime
FPS = 20
TRAINING_FPS = 180
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
MODEL_PATH = f"model/{MODEL_NAME}.pth"

MAX_MEMORY = 100_000
BATCH_SIZE = 1_000
LEARNING_RATE = 1e-3
HIDDEN_DIMENSIONS = [32]
GAMMA = 0.9

EPSILON_START = 1.0
EPSILON_DECAY = 0.9995
EPSILON_MIN = 0.01

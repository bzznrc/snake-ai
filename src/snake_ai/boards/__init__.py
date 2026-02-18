"""Board helpers for Bang AI."""

from .square_generation import spawn_connected_random_walk_shapes
from .square_specs import SQUARE_BOARD_STANDARD, SQUARE_CELL_RENDER_STANDARD

__all__ = [
    "SQUARE_BOARD_STANDARD",
    "SQUARE_CELL_RENDER_STANDARD",
    "spawn_connected_random_walk_shapes",
]

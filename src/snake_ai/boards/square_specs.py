"""Square-grid board presets."""

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class SquareBoardSpec:
    """Configuration for a board made of square cells."""

    columns: int
    rows: int
    cell_size_px: int
    bottom_bar_height_px: int
    bottom_bar_margin_px: int

    @property
    def playfield_width_px(self) -> int:
        return self.columns * self.cell_size_px

    @property
    def playfield_height_px(self) -> int:
        return self.rows * self.cell_size_px

    @property
    def screen_width_px(self) -> int:
        return self.playfield_width_px

    @property
    def screen_height_px(self) -> int:
        return self.playfield_height_px + self.bottom_bar_height_px


@dataclass(frozen=True)
class SquareCellRenderSpec:
    """Rendering tuning for nested square-cell visuals."""

    inset_px: int

    @property
    def inset_double_px(self) -> int:
        return self.inset_px * 2


SQUARE_BOARD_STANDARD: Final[SquareBoardSpec] = SquareBoardSpec(
    columns=32,
    rows=24,
    cell_size_px=20,
    bottom_bar_height_px=30,
    bottom_bar_margin_px=20,
)

SQUARE_CELL_RENDER_STANDARD: Final[SquareCellRenderSpec] = SquareCellRenderSpec(
    inset_px=4,
)

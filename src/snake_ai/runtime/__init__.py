"""Runtime helpers for Snake AI."""

from .helpers import configure_logging

try:
    from .arcade_runtime import ArcadeFrameClock, ArcadeWindowController, TextCache, load_font_once
except ModuleNotFoundError:
    pass

__all__ = [
    "ArcadeFrameClock",
    "ArcadeWindowController",
    "TextCache",
    "load_font_once",
    "configure_logging",
]

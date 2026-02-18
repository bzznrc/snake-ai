"""Small logging helpers for consistent console output."""

from __future__ import annotations

from collections import OrderedDict
import logging
from typing import Any


def configure_logging(level: str = "INFO") -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _format_mode(mode: str) -> str:
    words = mode.replace("-", " ").split()
    return " ".join("AI" if word.lower() == "ai" else word.title() for word in words)


def log_run_context(mode: str, context: dict[str, Any]) -> None:
    mode_label = _format_mode(mode)
    ordered = OrderedDict((key, value) for key, value in context.items() if value is not None)
    segments = [mode_label]
    segments.extend(f"{key.replace('_', ' ').title()}: {_format_value(value)}" for key, value in ordered.items())
    logging.getLogger("snake_ai.run").info(" / ".join(segments))

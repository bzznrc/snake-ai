"""Asset path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def _package_assets_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "assets"


def _iter_asset_roots(search_roots: Iterable[str | Path] | None):
    yield _package_assets_dir()

    if search_roots is not None:
        for root in search_roots:
            root_path = Path(root)
            yield root_path
            yield root_path / "assets"

    cwd = Path.cwd()
    yield cwd
    yield cwd / "assets"


def _resolve_asset_path(relative_path: str, search_roots: Iterable[str | Path] | None = None) -> str:
    raw_path = Path(relative_path)
    if raw_path.is_absolute() and raw_path.exists():
        return str(raw_path)

    normalized = relative_path.replace("\\", "/")
    for root in _iter_asset_roots(search_roots):
        candidate = root / normalized
        if candidate.exists():
            return str(candidate)

    return str(_package_assets_dir() / normalized)


def resolve_font_path(font_path_or_file: str, search_roots: Iterable[str | Path] | None = None) -> str:
    normalized = font_path_or_file.replace("\\", "/")
    if "/" not in normalized:
        normalized = f"fonts/{normalized}"
    return _resolve_asset_path(normalized, search_roots=search_roots)
